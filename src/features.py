import pandas as pd

# Cargar los datos desde data/interim
print("Cargando archivos desde data/interim...")
orders = pd.read_csv('../data/interim/orders.csv')
costs = pd.read_csv('../data/interim/costs.csv')
visits = pd.read_csv('../data/interim/visits.csv')

# Verificar las columnas cargadas
print("Columnas de orders.csv:", orders.columns)
print("Columnas de costs.csv:", costs.columns)
print("Columnas de visits.csv:", visits.columns)

# Convertir fechas a datetime
orders['buy_date'] = pd.to_datetime(orders['buy_date'], errors='coerce')
visits['session_start'] = pd.to_datetime(visits['start_ts'], errors='coerce')
costs['dt'] = pd.to_datetime(costs['dt'], errors='coerce')

# Variables de comportamiento
print("Generando variables de comportamiento...")
user_sources = visits[['uid', 'session_start', 'source_id']].drop_duplicates()
user_sources['session_date'] = user_sources['session_start'].dt.date
user_sources["session_date"] = pd.to_datetime(user_sources["session_date"])

# Número de sesiones por usuario
sessions_per_user = user_sources.groupby('uid')['session_date'].nunique().reset_index()
sessions_per_user.rename(columns={'session_date': 'num_sessions'}, inplace=True)

# Duración promedio y máxima de las sesiones
session_durations = visits.groupby('uid')['session_duration'].agg(['mean', 'max']).reset_index()
session_durations.rename(columns={'mean': 'avg_duration', 'max': 'max_duration'}, inplace=True)

# Fechas de la primera y última sesión
first_last_sessions = visits.groupby('uid').agg(
    first_session=('session_start', 'min'),
    last_session=('session_start', 'max')
).reset_index()

# Variables de comportamiento combinadas
user_behavior = pd.merge(sessions_per_user, session_durations, on='uid', how='left')
user_behavior = pd.merge(user_behavior, first_last_sessions, on='uid', how='left')

# Variables de marketing
print("Generando variables de marketing...")
# Guardar canal de adquisición (última fuente conocida del usuario)
last_source = user_sources.sort_values('session_start').groupby('uid').tail(1)
last_source = last_source[['uid', 'source_id']].rename(columns={'source_id': 'marketing_channel'})

# Etiquetas (targets)
print("Generando etiquetas (targets)...")
user_first_sessions = visits.groupby('uid')['session_start'].min().reset_index()
user_first_sessions.rename(columns={'session_start': 'first_session_date'}, inplace=True)

# Calcular LTV_180
orders = pd.merge(orders, user_first_sessions, on='uid', how='left')
orders['days_since_first_session'] = (orders['buy_date'] - orders['first_session_date']).dt.days
orders_180 = orders[orders['days_since_first_session'] <= 180]
ltv_180 = orders_180.groupby('uid')['revenue'].sum().reset_index()
ltv_180.rename(columns={'revenue': 'LTV_180'}, inplace=True)

# Calcular CAC_source_30
# Paso 1: identificar la fuente (source_id) desde la primera sesión de cada usuario
user_sources_first = user_sources.sort_values('session_start').drop_duplicates('uid', keep='first')
user_sources_first = user_sources_first[['uid', 'source_id', 'session_start']]
user_sources_first.rename(columns={'source_id': 'source_id_first', 'session_start': 'first_session_date'}, inplace=True)

# Paso 2: combinar con costs por source_id
cac_data = pd.merge(user_sources_first, costs, left_on='source_id_first', right_on='source_id', how='left')

# Paso 3: calcular días desde primera sesión
cac_data['days_since_first_session'] = (cac_data['dt'] - cac_data['first_session_date']).dt.days

# Paso 4: filtrar solo costos dentro de los primeros 30 días
cac_data_30 = cac_data[(cac_data['days_since_first_session'] >= 0) & (cac_data['days_since_first_session'] <= 30)]

# Paso 5: calcular CAC por usuario como costo promedio de su fuente en esos 30 días
cac_source_30 = cac_data_30.groupby('uid')['costs'].mean().reset_index()
cac_source_30.rename(columns={'costs': 'CAC_source_30'}, inplace=True)

# Combinamos todas las características
print("Combinando todas las características en un solo dataset...")
final_features = pd.merge(user_behavior, ltv_180, on='uid', how='left')
final_features = pd.merge(final_features, cac_source_30, on='uid', how='left')
final_features = pd.merge(final_features, last_source, on='uid', how='left')

# Guardar features
print("Guardando features en data/processed/features.csv...")
final_features.to_csv('../data/processed/features.csv', index=False)

print("¡Listo!")