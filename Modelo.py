"""
    Time series model: Outlier, level shift, and structural change detection

    The model estimation is performed using R. The python code retrieves the data and adjusts the inputs.
    The results, outliers and forecast, are stored on an external DB.
"""

########################################################################################################################
# #####################                Declare imports                   ####################### #
########################################################################################################################

import re
import time
import pypyodbc
import numpy as np
import pandas as pd
# import itertools as it
import multiprocessing as mp
from datetime import datetime
import Dashboard_Series.Funciones as fn

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages

import plotly.offline as py
import plotly.graph_objs as go

print('Imports OK')

########################################################################################################################
# #####################                Declare  'constants'          ####################### #
########################################################################################################################

path_folder = '//M006/'
path_plots = '//M006/Dashboard/Graphs/'
path_sql01 = path_folder + 'Queries/ret.sql'
path_sql02 = path_folder + 'Queries/dep.sql'
print('Constants OK')

########################################################################################################################
# #####################                declare dicts and 'private' functions          ####################### #
########################################################################################################################

cs_params = {'server': '', 'db': '', 'user': '', 'pass': ''}
cs_sql0 = 'Server=myServerAddress;Database=myDataBase;Trusted_Connection=True;;'
cs_sql = 'Server=' + cs_params['server']+ ';Database=' + cs_params['db'] +
        ';User Id=' + cs_params['user']+ ';Password=' + cs_params['pass'] + ';'

# pd.set_option('display.width', 200)
pd.options.display.float_format = '{:,.2f}'.format
rcParams['font.family'] = 'monospace'
rcParams['font.monospace'] = ['Courier New', 'Courier']

d_queries = {1: path_folder + 'Queries/ret.sql',
             2: path_folder + 'Queries/dep.sql',
             3: path_folder + 'Queries/impone.sql',
             4: path_folder + 'Queries/recibe.sql',
             5: path_folder + 'Queries/remite.sql',
             6: path_folder + 'Queries/recibe.sql',
             7: path_folder + 'Queries/venta.sql',
             8: path_folder + 'Queries/compra.sql'
             }

d_transaccion = {1: 'Tr_01', 2: 'Tr_02',
                 3: 'Tr_03', 4: 'Tr_04',
                 5: 'Tr_05', 6: 'Tr_06',
                 7: 'Tr_07', 8: 'Tr_08'}


l_colors = ['#0343df', '#f97306', '#15b01a', '#e50000', '#7e1e9c', '#653700', '#ff81c0', '#929591',
            '#6e750e', '#00ffff', '#c79fef', '#ff796c', '#01ff07', '#c04e01']


def milmillones_cop(x, pos):
    return '${:,.1f}'.format(x*1e-9), pos


formatomilmillonesCOP = FuncFormatter(milmillones_cop)


def millones_cop(x, pos):
    return '${:,.1f}'.format(x*1e-6), pos


formatomillonesCOP = FuncFormatter(millones_cop)


def billones_cop(x, pos):
    return '${:,.1f}'.format(x*1e-12), pos


formatobillonesCOP = FuncFormatter(billones_cop)


def milmillones(x, pos):
    return '{:,.1f}'.format(x*1e-9), pos


formatomilmillones = FuncFormatter(milmillones)


def millones(x, pos):
    return '{:,.1f}'.format(x*1e-6), pos


formatomillones = FuncFormatter(millones)


def billones(x, pos):
    return '{:,.1f}'.format(x*1e-12), pos


formatobillones = FuncFormatter(billones)


def fechames(x, pos):
    return pd.to_datetime([x], format='%Y%m'), pos
    # return '%Y - %m' % (x.year(), x.month())


formatofechames = FuncFormatter(fechames)


def select_exponente(expo):
    switcher = {
        6: "Millones de COP",
        9: "Miles de millones de COP",
        12: "Billones de COP"
    }
    return switcher.get(expo, "COP")


print('Funcs OK')

########################################################################################################################
# #####################                Data processing        ####################### #
########################################################################################################################
message = '\nIngrese el número de la opcción que desea correr:\n' + re.sub('\,', '\n', str(d_transaccion))
n_trans = int(input(message))
titulo_serie = d_transaccion[n_trans]
print("Ha seleccionado", titulo_serie, '- Opcion', n_trans, '\n')

# Loads the sql quey to retrieve the data
sql_command = open(d_queries[n_trans], 'r', encoding='latin1').read()
sql_command = re.sub('\ufeff|\n|\t', ' ', sql_command)
print(sql_command)

print('Start time: ', datetime.now().strftime('%H:%M:%S'))
global_start_time = time.time()

# Coonects to the DB and retrieves the data
con = pypyodbc.connect(cs_sql)
cursor = con.cursor()
cursor.execute(sql_command)
cols = cursor.description
cols = pd.DataFrame(cols)
out = cursor.fetchall()
print('query received ok')
out = pd.DataFrame(out)
out.columns = cols[0].tolist()
con.commit()
print('query commit')
print('Data retrieved. N=', len(out))
con.close()
out = out.sort_values(['fecha'])
out = out[out['fecha'] > 200900]
print('Data cleaned. N=', len(out))
data0 = out.copy()
print(data0.dtypes)

print("\nTiempo total: %s:%02d minutos" % divmod(int(time.time() - global_start_time), 60))
print('Finish time: ', datetime.now().strftime('%H:%M:%S'))

data0.to_csv(path_folder + '01-DataTimeline' + titulo_serie + '.csv', sep=';', index=False, encoding='latin1')
print('File saved')
"""
if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count()-1, maxtasksperchild=10)
    print('Pool created')
    # args = zip([d_queries[x] for x in range(1, len(d_queries) + 1)],
    #            [path_folder + 'Data_' + d_transaccion[x] for x in range(1, len(d_queries) + 1)])
    args = zip([d_queries[x] for x in range(1, 2)],
               [path_folder + 'Data_' + d_transaccion[x] for x in range(1, 2)])
    l_cols = pool.map_async(fn.f_outliers, args, chunksize=1)
    print('Pool map command sent')
    pool.close()
    print('Close commmand sent')
    pool.join()
    print('Join command sent')
"""

# --------------------------------------------------------------------------------------------
"""
    Dataframe data0 must be a table that contains:
    + fecha, (date) formatted YYYYMM,
    + lugar and codigo_lugar, or  codigo and nombre, or just nombre (name of the town considered)
    + valor, (value) transaction value in constant prices
"""
data0 = pd.read_csv(path_folder + '01-DataTimeline' + titulo_serie + '.csv', encoding='latin1', sep=';')
data0['valor'] = data0['valor'].astype('float')
# data0['valor'] = np.divide(data0['valor'], data0['ipc'])  # se deberia hace en el sql
data0['fecha_fmt'] = list(map(lambda x: x, pd.to_datetime(data0['fecha'], format='%Y%m')))  # str(x.date())
# data0['lugar'] = list(map(lambda x: re.sub("[0-9]|\(|\)", "", x), data0['lugar']))
data0['lugar'] = list(map(lambda x: re.sub("[0-9]|\(|\)", "", x), data0['municipio']))

# Create the dict of names and codes for the towns
# d_names = dict(zip(data0.codigo_lugar, data0.lugar))
d_names0 = dict(zip(data0.depto, data0.departamento))
d_names = dict(zip(data0.codigo_municipio,
                   ['{}, {}'.format(a, b) for (a, b) in zip(data0.municipio, data0.departamento)]))

deptos = data0[['depto', 'codigo_municipio']].sort_values(by=['depto', 'codigo_municipio']).drop_duplicates()
deptos = deptos.groupby('depto')['codigo_municipio'].apply(list)
d_mun = deptos.to_dict()

# -------------------------------------------------------------------------------------------------------------------- #
#   1. Data preparation
# -------------------------------------------------------------------------------------------------------------------- #
"""
Se genera una serie de dataframes para cada tipo de transacción.
El indice de la tabla es el mes, y las columnas son los municipios
    pvt: es la informacion original, en niveles.
    arma: es una copia de pvt
"""

# Debe venir separado: giros impone o recibe, efectivo retiros o depositos.
# No se hace filtro para dejar solo uno de los dos
pvt_df = data0.groupby(['codigo_municipio', 'fecha_fmt']).agg({'valor': 'sum'})
pvt_df = pvt_df.reset_index()
pvt_df = pvt_df.pivot(index='fecha_fmt', columns='codigo_municipio', values='valor')
arma_df = pvt_df.copy()
pvt_df.to_csv(path_folder + '02-Pivot' + titulo_serie + '.csv', sep=';', index=False, encoding='latin1')

# Guarda los exponentes de las cifras para simplificar la grafica
d_exp = dict(zip(arma_df.columns, np.floor(np.log10(np.max(arma_df, axis=0))/3)*3))
print('Data OK')

l_towns = [x for x in data0['codigo_municipio'].sort_values().unique()]
l_towns = [x for x in l_towns if arma_df[x].isnull().sum() < 2]
print("Se procesaran ", len(l_towns), " municipios.")

# -------------------------------------------------------------------------------------------------------------------- #
#   2. SARIMA estimation in R
# -------------------------------------------------------------------------------------------------------------------- #
"""
The R function is called, and objects are converted from pandas to R
Some diagnostic plots are created by the function
"""

# query_create_outlier = 'CREATE TABLE DASHBOARD_outliers(fecha_modelo character(8), municipio integer, ' \
#                        'tipo character(10), fecha_outlier character(6), indicador byteint, transaccion byteint);'
# query_create_forecast = 'CREATE TABLE DASHBOARD_forecast(fecha_modelo character(8), municipio integer, ' \
#                         'fecha_proyeccion character(6), valor numeric(19,4), upper numeric(19,4), ' \
#                         'lower numeric(19,4), transaccion byteint);'

print('Start time: ', datetime.now().strftime('%H:%M:%S'))
global_start_time = time.time()
# Limpia los archivos txt de validacion
file = open(r'C:\Users\saldana\PycharmProjects\untitled\DashboardEfectivo\Plots\Forecast_check.txt', 'w')
file.write('')
file.close()
file = open(r'C:\Users\saldana\PycharmProjects\untitled\DashboardEfectivo\Plots\Outliers_check.txt', 'w')
file.write('')
file.close()

arma_df.columns = list(map(lambda x: str(n_trans) + '_' + str(int(x)), arma_df.columns))
l_towns_ = list(map(lambda x: str(n_trans) + '_' + str(int(x)), l_towns))
if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count()-1, maxtasksperchild=10)
    print('Pool created')
    # with nostdout():
    pool.map_async(fn.f_outliers, arma_df[l_towns_].iteritems(), chunksize=1)
    print('Pool map command sent')
    pool.close()
    print('Close commmand sent')
    pool.join()
    print('Join command sent')

arma_df.columns = list(map(lambda x: int(str(x).split(sep='_')[1]), arma_df.columns))

print("\nTiempo total: %s:%02d minutos" % divmod(int(time.time() - global_start_time), 60))
print('Finish time: ', datetime.now().strftime('%H:%M:%S'))

# -------------------------------------------------------------------------------------------------------------------- #
#   3. Outliers and forecast data retrieval
# -------------------------------------------------------------------------------------------------------------------- #


# # ---------- Forecast -------------------------------------------------------------------
sql_command = 'SELECT municipio as lugar, cast(fecha_proyeccion as integer) fecha, valor, upper, lower ' \
              'FROM  DASHBOARD_forecast ' \
              'WHERE fecha_modelo = (select max(fecha_modelo) FROM  DASHBOARD_forecast)' \
              'AND transaccion = ' + str(n_trans) + ';'
print(sql_command)

print('Start time: ', datetime.now().strftime('%H:%M:%S'))
global_start_time = time.time()
# Connect to DB
con = pypyodbc.connect(cs_sql)
cursor = con.cursor()
cursor.execute(sql_command)
cols = cursor.description
cols = pd.DataFrame(cols)
out = cursor.fetchall()
print('query received ok')
out = pd.DataFrame(out)
out.columns = cols[0].tolist()
con.commit()
print('query commit')
print('Data retrieved. N=', len(out))
con.close()
out = out.sort_values(['fecha'])
out = out[out['fecha'] > 200900]
print('Data cleaned. N=', len(out))
data0 = out.copy()
print("\nTiempo total: %s:%02d minutos" % divmod(int(time.time() - global_start_time), 60))
print('Finish time: ', datetime.now().strftime('%H:%M:%S'))

data0['fecha'] = list(map(lambda x: x, pd.to_datetime(data0['fecha'], format='%Y%m')))
data0['valor'] = data0['valor'].astype('float')
data0['lower'] = data0['lower'].astype('float')
data0['upper'] = data0['upper'].astype('float')
df_forecast = data0.pivot(index='fecha', columns='lugar', values='valor')
df_forecast_l = data0.pivot(index='fecha', columns='lugar', values='lower')
df_forecast_u = data0.pivot(index='fecha', columns='lugar', values='upper')
print('DF_forecast OK')
"""
l_frames = []
for fc_type in ['valor', 'lower', 'upper']:
    tmp = data0[['fecha', 'lugar', fc_type]]
    tmp['tipo'] = fc_type
    l_frames.append(tmp.copy())
df_forecast_all = pd.DataFrame(l_frames)
df_forecast_all = df_forecast_all.pivot(index=['fecha', 'tipo'], columns='lugar', values='valor')
"""

# # ---------- Outliers -------------------------------------------------------------------
sql_command = 'SELECT tipo, cast(fecha_outlier as integer) fecha, municipio as lugar, indicador ' \
              'FROM  DASHBOARD_outliers ' \
              'WHERE fecha_modelo = (select max(fecha_modelo) FROM  DASHBOARD_outliers)' \
              'AND transaccion = ' + str(n_trans) + ';'
print(sql_command)

con = pypyodbc.connect(cs_sql)
cursor = con.cursor()
cursor.execute(sql_command)
cols = cursor.description
cols = pd.DataFrame(cols)
out = cursor.fetchall()
print('query received ok')
out = pd.DataFrame(out)
out.columns = cols[0].tolist()
con.commit()
print('query commit')
print('Data retrieved. N=', len(out))
con.close()
out = out.sort_values(['fecha'])
out = out[out['fecha'] > 200900]
out = out.drop_duplicates()
print('Data cleaned. N=', len(out))
data0 = out.copy()

data0['tipo'] = list(map(lambda x: x.strip(), data0['tipo']))
data0['fecha'] = list(map(lambda x: x, pd.to_datetime(data0['fecha'], format='%Y%m')))
df_outliers_ao = pd.DataFrame(index=pvt_df.index, columns=df_forecast.columns).fillna(0)
df_outliers_ls = df_outliers_ao.copy()
df_breaks = df_outliers_ao.copy()
if len(data0) > 0:
    for mun in data0[data0['tipo'] == 'AO']['lugar'].unique():
        df_outliers_ao.loc[df_outliers_ao.index.isin(list(data0[np.logical_and(data0['tipo'] == 'AO',
                                                                               data0['lugar'] == mun)]['fecha'])),
                           mun] = list(data0[np.logical_and(data0['tipo'] == 'AO',
                                                            data0['lugar'] == mun)]['indicador'])
    for mun in data0[data0['tipo'] == 'LS']['lugar'].unique():
        df_outliers_ls.loc[df_outliers_ls.index.isin(list(data0[np.logical_and(data0['tipo'] == 'LS',
                                                                               data0['lugar'] == mun)]['fecha'])),
                           mun] = list(data0[np.logical_and(data0['tipo'] == 'LS',
                                                            data0['lugar'] == mun)]['indicador'])
    for mun in data0[data0['tipo'] == 'BP']['lugar'].unique():
        df_breaks.loc[df_breaks.index.isin(list(data0[np.logical_and(data0['tipo'] == 'BP',
                                                                     data0['lugar'] == mun)]['fecha'])),
                      mun] = list(data0[np.logical_and(data0['tipo'] == 'BP',
                                                       data0['lugar'] == mun)]['indicador'])
print('DF_outliers OK')
# todo use dict to separate outliers by type in summary pdf file
"""
df_outliers = data0.copy()
df_outliers['tipo_dato'] = list(map(lambda x: fn.d_outl(x[0], x[1]), zip(data0['coefhat'], data0['type'])))
df_outliers = df_outliers.pivot(index='fecha', columns='lugar', values='tipo_dato')

fn.plot_summary(df_raw=arma_df, df_fc=df_forecast, df_outl=df_outliers,
                col_names=list(map(lambda x: d_names[x], arma_df.columns)),
                path=path_folder, name_df=titulo_serie)
"""

# -------------------------------------------------------------------------------------------------------------------- #
#   4. Plot outlier position/date
# -------------------------------------------------------------------------------------------------------------------- #
for dept in d_mun.keys():
    l_towns = d_mun[dept]
    # print('****', dept)
    # print(l_towns)
    # Series -----------------------------------------------------------------------------------------------------------
    annotations0 = list([dict(x=0, y=-0.3, showarrow=False,  text='<b>Fuente:</b>', visible=True, xref='paper',
                              yref='paper', font=dict(family='Courier New', size=18)),
                         dict(x=0, y=1.12, showarrow=False,  text='<i>Municipio</i>', visible=True, xref='paper',
                              yref='paper', font=dict(family='Courier New', size=16)),
                         ])

    botones = list([])
    for n, k in enumerate(l_towns):
        # valm = max(arma_df[k])/(10**d_exp[k])
        botones.append(dict(args=[{'visible': [x == k for x in l_towns]*8},
                                  {'annotations': list([dict(x=-0.07, y=0.5, xref='paper', yref='paper',
                                                        font=dict(family='Courier New', size=18, color='rgb(0,128,0)'),
                                                        text=select_exponente(d_exp[k]), visible=True, showarrow=False,
                                                        textangle=-90)]) + annotations0}
                                  ],
                            label=d_names[k], method='update'))

    updatemenu = list([
        dict(
            buttons=botones, direction='down', showactive=True, pad={'r': 15, 't': 10}, x=0, y=1.1,
            xanchor='left', yanchor='top', bgcolor='#FFFFFF', bordercolor='#000000',
            font=dict(family='Courier New', size=12)
        )
    ])

    layout = go.Layout(title='<b>Linea de tiempo mensual de ' + titulo_serie + ' en ' + d_names0[dept] +
                             '</b><br>Valores en pesos constantes<br><br>',
                       titlefont=dict(family='Courier New', size=30),
                       showlegend=True,
                       xaxis=dict(title='Fecha', range=[str(min(arma_df.index.values[-20:]).astype('<M8[D]')),
                                                        str(max(df_forecast.index.values).astype('<M8[D]'))],
                                  tick0='2009-01-01', autorange=False,
                                  rangeslider={'autorange': False,
                                               'range': [min(arma_df.index), max(df_forecast.index)]},
                                  titlefont=dict(family='Courier New', size=18), tickfont=dict(family='Courier New'),
                                  type='date'),
                       yaxis=dict(title='Número de outliers', autorange=False,  tickformat=',.1f', fixedrange=True,
                                  range=[0, 5], rangemode='nonnegative', titlefont=dict(family='Courier New', size=18),
                                  side='right', tickfont=dict(family='Courier New', color='rgb(0,0,139)')),
                       yaxis2=dict(title='', autorange=True,  tickformat='$,.2f', rangemode='nonnegative',
                                   fixedrange=False, overlaying='y', side='left', showgrid=False,
                                   titlefont=dict(family='Courier New', size=18, color='rgb(0,128,0)'),
                                   tickfont=dict(family='Courier New', color='rgb(0,128,0)')),
                       margin=dict(pad=5, t=130, r=60, l=120, autoexpand=True),
                       annotations=annotations0 + [dict(x=-0.07, y=0.5, xref='paper', yref='paper', showarrow=False,
                                                        font=dict(family='Courier New', size=18, color='rgb(0,128,0)'),
                                                        text=select_exponente(d_exp[l_towns[0]]), visible=True,
                                                        textangle=-90)],
                       legend=dict(bgcolor='rgb(255,255,255)', bordercolor='rgb(0,0,0)', borderwidth=1, x=1.12, y=0.5,
                                   font=dict(family='Courier New'), traceorder='grouped'),
                       barmode='stack',
                       paper_bgcolor='rgba(255,255,255,0)',
                       plot_bgcolor='rgba(255,255,255,1)',
                       # shapes=l_fechas,
                       updatemenus=updatemenu
                       )
    datos = list([])
    # # timeseries
    for k in l_towns:
        datos.append(go.Scatter(y=list(np.divide(arma_df[k], 10**d_exp[k])),
                                x=arma_df.index,
                                name='Serie original', visible=(k == l_towns[0]), marker=dict(color='rgb(0,128,0)'),
                                legendgroup='Series', yaxis='y2'
                                ))
    # # Moving average
    for k in l_towns:
        datos.append(go.Scatter(y=list(np.divide(arma_df[k].rolling(window=12, min_periods=6).mean(), 10**d_exp[k])),
                                x=arma_df.index,
                                name='Media 12 meses', visible=(k == l_towns[0]), marker=dict(color='rgb(50,205,50)'),
                                legendgroup='Series', yaxis='y2'
                                ))
    # # Forecast -mean
    for k in l_towns:
        if k in [x for x in l_towns if x in df_forecast.columns]:
            datos.append(go.Scatter(y=list(np.divide(df_forecast[k], 10**d_exp[k])),
                                    x=df_forecast.index,
                                    name='Proyección', visible=(k == l_towns[0]), marker=dict(color='rgb(75,0,130)'),
                                    legendgroup='forecast', yaxis='y2'))
        else:
            datos.append(go.Scatter(y=list([np.nan]*len(df_forecast)),
                                    x=df_forecast.index,  # 'Intervalo de confianza (95%)'
                                    name='Proyección', visible=(k == l_towns[0]), marker=dict(color='rgb(75,0,130)'),
                                    legendgroup='forecast', yaxis='y2'))
    # # Forecast -lower limit
    for k in l_towns:
        if k in [x for x in l_towns if x in df_forecast_l.columns]:
            datos.append(go.Scatter(y=list(np.divide(df_forecast_l[k], 10**d_exp[k])),
                                    x=df_forecast_l.index,  # 'Intervalo de confianza (95%)'
                                    name='Intervalo de confianza (95%)',
                                    visible=(k == l_towns[0]), yaxis='y2', mode='lines',
                                    showlegend=False, marker=dict(color='rgb(143,19,131)'), legendgroup='forecast'))
        else:
            datos.append(go.Scatter(y=list([np.nan]*(len(df_forecast_l))),
                                    x=df_forecast_l.index,  # 'Intervalo de confianza (95%)'
                                    name='Intervalo de confianza (95%)',
                                    visible=(k == l_towns[0]), yaxis='y2', mode='lines',
                                    showlegend=False, marker=dict(color='rgb(143,19,131)'), legendgroup='forecast'))
    # # Forecast -upper limit
    for k in l_towns:
        if k in [x for x in l_towns if x in df_forecast_u.columns]:
            datos.append(go.Scatter(y=list(np.divide(df_forecast_u[k], 10**d_exp[k])),
                                    x=df_forecast_u.index,
                                    name='Intervalo de confianza (95%)',
                                    visible=(k == l_towns[0]), yaxis='y2', fill='tonexty',
                                    showlegend=True, marker=dict(color='rgb(143,19,131)'), legendgroup='forecast',
                                    mode='lines'))
        else:
            datos.append(go.Scatter(y=list([np.nan]*(len(df_forecast_u))),
                                    x=df_forecast_u.index,  # 'Intervalo de confianza (95%)'
                                    name='Intervalo de confianza (95%)',
                                    visible=(k == l_towns[0]), yaxis='y2', mode='lines',
                                    showlegend=False, marker=dict(color='rgb(143,19,131)'), legendgroup='forecast'))
    # # Outliers AO
    for k in l_towns:
        if k in [x for x in l_towns if x in df_outliers_ao.columns]:
            datos.append(go.Bar(y=list(np.abs(df_outliers_ao[k])),
                                x=df_outliers_ao.index,
                                name='Dato atípico',  visible=(k == l_towns[0]), marker=dict(color='rgb(106,90,205)'),
                                legendgroup='Outliers', hoverinfo='none'))
        else:
            datos.append(go.Bar(y=list(np.zeros(len(df_outliers_ao))),
                                x=df_outliers_ao.index,
                                name='Dato atípico',  visible=(k == l_towns[0]), marker=dict(color='rgb(106,90,205)'),
                                legendgroup='Outliers', hoverinfo='none'))
    # # Outliers LS
    for k in l_towns:
        if k in [x for x in l_towns if x in df_outliers_ls.columns]:
            datos.append(go.Bar(y=list(np.abs(df_outliers_ls[k])),
                                x=df_outliers_ls.index,
                                name='Cambio de tendencia',
                                visible=(k == l_towns[0]), marker=dict(color='rgb(0,0,139)'),
                                legendgroup='Outliers', hoverinfo='none'))
        else:
            datos.append(go.Bar(y=list(np.zeros(len(df_outliers_ao))),
                                x=df_outliers_ao.index,
                                name='Cambio de tendencia',
                                visible=(k == l_towns[0]), marker=dict(color='rgb(0,0,139)'),
                                legendgroup='Outliers', hoverinfo='none'))
    # # Breaks
    for k in l_towns:
        if k in [x for x in l_towns if x in df_breaks.columns]:
            datos.append(go.Bar(y=list(df_breaks[k]),
                                x=df_breaks.index,
                                name='Cambio estructural',
                                visible=(k == l_towns[0]), marker=dict(color='rgb(30,144,255)'),
                                legendgroup='Outliers', hoverinfo='none'))
        else:
            datos.append(go.Bar(y=list(np.zeros(len(df_outliers_ao))),
                                x=df_outliers_ao.index,
                                name='Cambio estructural',
                                visible=(k == l_towns[0]), marker=dict(color='rgb(30,144,255)'),
                                legendgroup='Outliers', hoverinfo='none'))

    plotdata = go.Data(datos)
    fig1 = go.Figure(data=plotdata, layout=layout)
    py.plot(fig1, filename=path_plots + titulo_serie + '/D_outliers_' + str(dept) + '.html', auto_open=False)

# -------------------------------------------------------------------------------------------------------------------- #
#   5. html construction
# -------------------------------------------------------------------------------------------------------------------- #

# Writes an html file that joins all groups by transaction type
html_file0 = open(path_folder + '/Dashboard/Utils/Template_transaccion.html', 'r', encoding='latin1').read()
html_file = re.sub('<!-- @XYZ -->', titulo_serie, html_file0)
div_tag = ''
for dept in d_mun.keys():
    div_tag = div_tag + \
              fn.f_htmltag(d_names0[dept], path_plots + titulo_serie + '/' + 'D_outliers_' + str(dept) + '.html')
html_file = re.sub('<!-- @ABCD -->', div_tag, html_file)
default_plot = fn.f_htmltag2(path_plots + titulo_serie + '/' + 'D_outliers_' + str(dept) + '.html')
html_file = re.sub('<!-- @EFGH -->', default_plot, html_file)
file = open(path_plots + 'Frame_' + str(n_trans) + '_' + titulo_serie + '.html', 'w', encoding='utf8')
file.write(html_file)
file.close()
print('HTML ok')

# -------------------------------------------------------------------------------------------------------------------- #
#   6. summary file
# -------------------------------------------------------------------------------------------------------------------- #

pp = PdfPages(path_plots + 'ResumenOutliers' + titulo_serie + '.pdf')
metadata = pp.infodict()
metadata['Title'] = 'Resumen de los outliers'
metadata['Author'] = 'SAE, UIAF'
metadata['Subject'] = 'Resumen gráfico de los datos atípicos en ' + titulo_serie

fig = plt.figure(0, figsize=(12, 6))
plt.text(.5, .6, 'Resumen de los datos atípicos\n', fontsize=34, va='center', ha='center')
plt.text(.5, .6, '\n' + titulo_serie, fontsize=30, va='center', ha='center')
plt.text(.5, .35, '\n\n\n\nUIAF', fontsize=30, va='center', ha='center')
plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
plt.savefig(pp, format='pdf')
plt.close()

# Info agg by date
plt.figure(0, figsize=(12, 6))
plt.fill_between(df_breaks.index,
                 np.sum(np.abs(df_outliers_ao), axis=1) + np.sum(np.abs(df_outliers_ls), axis=1),
                 np.sum(np.abs(df_outliers_ao), axis=1) + np.sum(np.abs(df_outliers_ls), axis=1) +
                 np.sum(np.abs(df_breaks), axis=1),
                 label='Cambio estructural', color=l_colors[2], step='mid', alpha=1, lw=0)
plt.fill_between(df_outliers_ls.index,
                 np.sum(np.abs(df_outliers_ao), axis=1),
                 np.sum(np.abs(df_outliers_ao), axis=1) + np.sum(np.abs(df_outliers_ls), axis=1),
                 label='Cambio de tendencia', color=l_colors[1], step='mid', alpha=1, lw=0)
plt.fill_between(df_outliers_ao.index,
                 0,
                 np.sum(np.abs(df_outliers_ao), axis=1),
                 label='Dato Atípico', color=l_colors[0], step='mid', alpha=1, lw=0)
# plt.fill_between(df_outliers_ao.index, 0, 0, label=None, color='k', step='mid')
plt.axes().format_xdata = mdates.DateFormatter('%Y-%m')
plt.xlabel('Fecha')
plt.suptitle('Frecuencia de los outliers por fecha', fontsize=18)
plt.title(titulo_serie)
# plt.plot((np.nan, np.nan), (np.nan, np.nan), 'r--', label='Elecciones Presidenciales', linewidth=3)
# plt.plot((np.nan, np.nan), (np.nan, np.nan), 'g--', label='Elecciones Regionales', linewidth=3)
years = mdates.YearLocator()
# plt.axes().yaxis.set_major_formatter(formatomillones)
plt.ylabel('Frecuencia')
plt.axes().yaxis.grid(True)
plt.axes().set_ylim(bottom=0)
plt.yticks(rotation=0, wrap=False, ha='right')
plt.xticks(rotation=0, wrap=False, ha='center')
plt.figtext(0, 0, 'Fuente: \n', fontweight='bold')
plt.subplots_adjust(right=0.75, left=0.1)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=False, ncol=1)
plt.axes().xaxis.set_major_locator(years)
plt.savefig(pp, format='pdf')
plt.close()

# Towns with the most outliers
dataset = list(pd.Series(np.sum(np.add(np.abs(df_outliers_ao), np.add(np.abs(df_outliers_ls), np.abs(df_breaks))),
                                axis=0)).sort_values(ascending=False).head(15).index)
df_outliers_ao = df_outliers_ao[dataset]
df_outliers_ls = df_outliers_ls[dataset]
df_breaks = df_breaks[dataset]

plt.figure(0, figsize=(12, 6))
plt.bar(left=np.arange(0, len(df_outliers_ao.columns)), height=np.sum(np.abs(df_outliers_ao), axis=0),
        width=0.9, label='Dato Atípico', color=l_colors[0])
plt.bar(left=np.arange(0, len(df_outliers_ls.columns)), bottom=np.sum(np.abs(df_outliers_ao), axis=0),
        height=np.sum(np.abs(df_outliers_ls), axis=0), width=0.9, label='Cambio de tendencia',
        color=l_colors[1])
plt.bar(left=np.arange(0, len(df_breaks.columns)), height=np.sum(np.abs(df_breaks), axis=0),
        bottom=np.sum(np.abs(df_outliers_ao), axis=0)+np.sum(np.abs(df_outliers_ls), axis=0), width=0.9,
        label='Cambio estructural', color=l_colors[2])
plt.xticks(np.arange(0, len(df_breaks.columns)),
           list(map(lambda x: re.sub(', ', ',\n', d_names[x]), df_breaks.columns)))
plt.xlabel('Municipios')
plt.suptitle('Municipios con más outliers', fontsize=18)
plt.title(titulo_serie)
plt.ylabel('Frecuencia')
plt.axes().yaxis.grid(True)
plt.yticks(rotation=0, wrap=False, ha='right')
plt.xticks(rotation=90, wrap=True, ha='right', va='center')
plt.figtext(0, 0, 'Fuente: \n', fontweight='bold')
plt.subplots_adjust(right=0.75, left=0.1, bottom=0.2)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=False, ncol=1)
plt.savefig(pp, format='pdf')
plt.close()

# Close pdf
pp.close()

# -------------------------------------------------------------------------------------------------------------------- #
#   End of file
# -------------------------------------------------------------------------------------------------------------------- #
