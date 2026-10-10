import sys
from datetime import datetime, timedelta, timezone

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.types import StructType
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, regexp_extract, substring, when, round, coalesce, lit, row_number, desc
)
from awsglue.dynamicframe import DynamicFrame


def campo_dataType_ci(struct_type, nombre):
    for campo in struct_type.fields:
        if campo.name.lower() == nombre.lower():
            return campo.dataType
    raise KeyError(f"No se encontro el campo '{nombre}' (case-insensitive) en el struct")


def extraer_value_numerico(df, path_struct, alias):
    tipo_actual = df.schema
    for parte in path_struct.split("."):
        tipo_actual = campo_dataType_ci(tipo_actual, parte)
    tipo_value = campo_dataType_ci(tipo_actual, "value")

    if isinstance(tipo_value, StructType):
        return coalesce(
            col(f"{path_struct}.value.double"),
            col(f"{path_struct}.value.int").cast("double")
        ).alias(alias)
    else:
        return col(f"{path_struct}.value").cast("double").alias(alias)


def parse_observaciones(df):
    campos = [
        ("temperature", "obs_temp_c"),
        ("relativehumidity", "obs_rh_pct"),
        ("windspeed", "obs_windspeed_kmh"),
        ("dewpoint", "obs_dewpoint_c"),
        ("heatindex", "obs_heat_index_c"),
        ("winddirection", "obs_wind_direction_d"),
    ]
    selects = [
        col("station_id").alias("obs_station_id"),
        col("timestamp").alias("obs_timestamp"),
        substring(col("timestamp"), 1, 13).alias("join_time_key"),
    ]
    selects += [extraer_value_numerico(df, campo, alias) for campo, alias in campos]

    parsed_df = df.select(*selects)
    parsed_df = parsed_df.filter(col("obs_station_id").isNotNull())

    # Dedupe: si una estacion no actualizo su sensor entre dos corridas horarias
    # consecutivas, /observations/latest devuelve la MISMA observacion (mismo
    # timestamp) en dos particiones bronze distintas. En modo backfill (que lee
    # la tabla completa) esto se traduce en filas duplicadas con el mismo
    # join_time_key, que despues duplican el resultado del join. Nos quedamos
    # con una sola fila por estacion+hora, la de timestamp mas reciente.
    ventana = Window.partitionBy("obs_station_id", "join_time_key").orderBy(desc("obs_timestamp"))
    parsed_df = (
        parsed_df.withColumn("_rn", row_number().over(ventana))
        .filter(col("_rn") == 1)
        .drop("_rn")
    )
    return parsed_df


def parse_pronosticos(df):
    extracted_df = df.select(
        col("station_id").alias("fcst_station_id"),
        col("valido_desde_utc"),
        col("fecha_extraccion_utc").alias("fcst_generated_at"),
        col("horizonte_horas").cast("int").alias("horizonte_horas"),
        substring(col("valido_desde_utc"), 1, 13).alias("join_time_key"),
        col("forecast.temperature").cast("double").alias("fcst_temp_raw"),
        col("forecast.temperatureunit").alias("fcst_temp_unit"),
        extraer_value_numerico(df, "forecast.probabilityofprecipitation", "fcst_pop_pct"),
        extraer_value_numerico(df, "forecast.relativehumidity", "fcst_rh_pct"),
        extraer_value_numerico(df, "forecast.dewpoint", "fcst_dewpoint_c"),
        regexp_extract(col("forecast.windspeed"), r"([\d.-]+)", 1).cast("double").alias("fcst_wind_mph")
    )

    transformed_df = extracted_df.withColumn(
        "fcst_temp_c",
        when(col("fcst_temp_unit") == "F", round((col("fcst_temp_raw") - 32) * 5/9, 2))
        .otherwise(col("fcst_temp_raw"))
    ).withColumn(
        "fcst_windspeed_kmh",
        round(col("fcst_wind_mph") * 1.60934, 2)
    ).drop("fcst_temp_raw", "fcst_temp_unit", "fcst_wind_mph")

    return transformed_df.filter(col("fcst_station_id").isNotNull())


def join_and_select(obs_df, fcst_df):
    joined_df = obs_df.join(
        fcst_df,
        (obs_df.obs_station_id == fcst_df.fcst_station_id) &
        (obs_df.join_time_key == fcst_df.join_time_key),
        "inner"
    )
    final_df = joined_df.select(
        obs_df.obs_station_id.alias("station_id"),
        obs_df.join_time_key.alias("target_hour_utc"),
        col("horizonte_horas"),
        col("obs_timestamp"),
        col("valido_desde_utc").alias("fcst_target_timestamp"),
        col("fcst_generated_at"),
        col("obs_temp_c"),
        col("fcst_temp_c"),
        col("obs_heat_index_c"),
        col("obs_rh_pct"),
        col("fcst_rh_pct"),
        col("obs_wind_direction_d"),
        col("obs_windspeed_kmh"),
        col("fcst_windspeed_kmh"),
        col("obs_dewpoint_c"),
        col("fcst_dewpoint_c"),
        col("fcst_pop_pct")
    )
    return final_df


def calcular_predicate_ventana_pronosticos(target_dt, ventana_horas=24):
    condiciones = []
    for h in range(0, ventana_horas + 1):
        dt = target_dt - timedelta(hours=h)
        condiciones.append(
            f"(year='{dt.year}' and month='{dt.month:02d}' "
            f"and day='{dt.day:02d}' and hour='{dt.hour:02d}')"
        )
    return " or ".join(condiciones)


def optional_arg(argv, name):
    flag = f"--{name}"
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def main():
    args = getResolvedOptions(sys.argv, ['JOB_NAME'])

    sc = SparkContext()
    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(args['JOB_NAME'], args)

    # dynamic overwrite reemplaza SOLO las combinaciones de columnas de particiÃ³n
    # presentes en el DataFrame que se escribe -- por eso el particionado de
    # salida (fecha_particion + hora_particion, ver mas abajo) tiene que llegar
    # hasta el mismo grano que se produce en modo single-hour, o se termina
    # borrando de mas (bug corregido en esta version).
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    target_date = optional_arg(sys.argv, 'target_date')
    target_hour = optional_arg(sys.argv, 'target_hour')

    if target_date and target_hour:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").replace(hour=int(target_hour), tzinfo=timezone.utc)
        obs_predicate = (
            f"year='{target_dt.year}' and month='{target_dt.month:02d}' "
            f"and day='{target_dt.day:02d}' and hour='{target_dt.hour:02d}'"
        )
        fcst_predicate = calcular_predicate_ventana_pronosticos(target_dt, ventana_horas=24)
        print(f"Modo single-hour. obs_predicate={obs_predicate}")

        dynamic_frame_obs = glueContext.create_dynamic_frame.from_catalog(
            database="db-grupo-2", table_name="observaciones",
            transformation_ctx="datasource_observaciones", push_down_predicate=obs_predicate
        )
        dynamic_frame_pronosticos = glueContext.create_dynamic_frame.from_catalog(
            database="db-grupo-2", table_name="pronosticos",
            transformation_ctx="datasource_pronosticos", push_down_predicate=fcst_predicate
        )
    else:
        print("Modo backfill: sin target_date/target_hour, se procesa el historico completo.")
        dynamic_frame_obs = glueContext.create_dynamic_frame.from_catalog(
            database="db-grupo-2", table_name="observaciones",
            transformation_ctx="datasource_observaciones"
        )
        dynamic_frame_pronosticos = glueContext.create_dynamic_frame.from_catalog(
            database="db-grupo-2", table_name="pronosticos",
            transformation_ctx="datasource_pronosticos"
        )

    df_obs_raw = dynamic_frame_obs.toDF()
    df_pron_raw = dynamic_frame_pronosticos.toDF()

    # .isEmpty() es nativo del DataFrame desde Spark 3.3 -- evita forzar la
    # conversion a RDD que hacia .rdd.isEmpty(), que desactiva optimizaciones
    # de Catalyst y agrega overhead de serializacion.
    if df_obs_raw.isEmpty() or df_pron_raw.isEmpty():
        print("Sin datos disponibles. Nada para procesar.")
        job.commit()
        return

    df_obs_parsed = parse_observaciones(df_obs_raw)
    df_pron_parsed = parse_pronosticos(df_pron_raw)

    if target_date and target_hour:
        target_join_key = target_dt.strftime("%Y-%m-%dT%H")
        df_pron_parsed = df_pron_parsed.filter(col("join_time_key") == target_join_key)

    if df_pron_parsed.isEmpty():
        print("No hay pronosticos que matcheen. Nada para procesar.")
        job.commit()
        return

    df_silver = join_and_select(df_obs_parsed, df_pron_parsed)

    # Particion a nivel dia + hora (no solo dia): el overwrite dinamico borra
    # toda combinacion de columnas de particion presente en el DataFrame que
    # se escribe. Si solo particionaramos por fecha_particion, una corrida en
    # modo single-hour (que solo trae UNA hora) borraria el DIA COMPLETO ya
    # escrito por corridas anteriores, dejando solo esa hora. Particionando
    # tambien por hora, el overwrite dinamico solo reemplaza esa hora puntual.
    df_silver = df_silver.withColumn("fecha_particion", substring(col("target_hour_utc"), 1, 10))
    df_silver = df_silver.withColumn("hora_particion", substring(col("target_hour_utc"), 12, 2))

    df_silver.write.mode("overwrite").partitionBy("fecha_particion", "hora_particion").parquet(
        "s3://clima-dataset-grupo2/silver/clima_consolidado/"
    )

    job.commit()


if __name__ == '__main__':
    main()