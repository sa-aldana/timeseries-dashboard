"""
    Time series model - Functions
"""

########################################################################################################################
# #####################                Declare Imports                    ####################### #
########################################################################################################################

import sys
import contextlib

import re
import os
import time
import pypyodbc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter
from matplotlib import rcParams
# import matplotlib.ticker as ticker
# from statsmodels.tsa.statespace import sarimax


########################################################################################################################
# #####################                Declare common objects        ####################### #
########################################################################################################################

d_outliers = {1: (1, 'AO'), 2: (-1, 'AO'),
              3: (1, 'LS'), 4: (-1, 'LS'),
              5: (1, 'BP')}

########################################################################################################################
# #####################                Declare dicts and  'private' functions          ####################### #
########################################################################################################################


class DummmyFile(object):
    def write(self, x): pass

    def flush(self): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummmyFile()
    yield
    sys.stdout = save_stdout


def milmillonesCOP(x, pos):
    return '${:,.1f}'.format(x*1e-9)


formatomilmillonesCOP = FuncFormatter(milmillonesCOP)


def millonesCOP(x, pos):
    return '${:,.1f}'.format(x*1e-6)


formatomillonesCOP = FuncFormatter(millonesCOP)


def billonesCOP(x, pos):
    return '${:,.1f}'.format(x*1e-12)


formatobillonesCOP = FuncFormatter(billonesCOP)


def milmillones(x, pos):
    return '{:,.1f}'.format(x*1e-9)


formatomilmillones = FuncFormatter(milmillones)


def millones(x, pos):
    return '{:,.1f}'.format(x*1e-6)


formatomillones = FuncFormatter(millones)


def billones(x, pos):
    return '{:,.1f}'.format(x*1e-12)


formatobillones = FuncFormatter(billones)


def fechames(x, pos):
    return pd.to_datetime([x], format='%Y%m')
    # return '%Y - %m' % (x.year(), x.month())


formatofechames = FuncFormatter(fechames)

########################################################################################################################
# #####################                declare 'constant' values                    ####################### #
########################################################################################################################

cs_params = {'server': '', 'db': '', 'user': '', 'pass': ''}
cs_sql0 = 'Server=myServerAddress;Database=myDataBase;Trusted_Connection=True;;'
cs_sql = 'Server=' + cs_params['server']+ ';Database=' + cs_params['db'] +
        ';User Id=' + cs_params['user']+ ';Password=' + cs_params['pass'] + ';'


########################################################################################################################
# #####################                Declare 'public' functions                    ####################### #
########################################################################################################################


def f_outliers(tupla) -> None:
    """
    Receives a tuple that contains a time series, calls the Sarima function (in R) which adjusts a sarima model
    and tests for outliers, level shifts and structural changes.

    :param tupla: tup that contains the timeseries and its name/code
    :return: None. Results are saved in a sql DB.
    """
    t_mun, serie = tupla
    trans = int(str(t_mun).split(sep='_')[0])
    mun = int(str(t_mun).split(sep='_')[1])
    print(os.getpid(), " iniciando ", mun)
    pandas2ri.activate()
    with nostdout():
        plot_path = 'C:\\Users\\saldana\\PycharmProjects\\untitled\\Dashboard_Series\\Plots\\'
        ro.r('''source('C:/Users/saldana/PycharmProjects/untitled/Dashboard_Series/SARIMA.R')''')
        sarima = ro.r['est_sarima']
        serie = pd.Series(serie)
    # print("stdout 1 ok")
    if serie.count() > len(serie.index)-1:
        query_append = ""
        # with nostdout():
        ans = sarima(serie=ro.FloatVector(serie), nombre='df_' + str(int(mun)),
                     startYear=serie.index[0].year, startMonth=serie.index[0].month)
        ro.r('''dev.off()''')
        # print("stdout 2 ok")
        print(mun, '\t', len(ans[2]), len(ans[4]), len(ans[6]))
        # print(mun, '\t', len(ans[2]) > 0)
        # # Stores the outliers' positions
        if len(ans[2]) > 0:
            outliers = pandas2ri.ri2py(ans[2])
            outliers['f_model'] = time.strftime('%Y%m%d')
            outliers['ind'] = list(map(lambda x: pd.to_datetime(str(int(x)), format='%Y%m'), outliers['time']))
            outliers['mun'] = mun
            outliers['trans'] = trans
            for n in range(len(outliers)):
                query_append += "\nINSERT INTO DASHBOARD_outliers (fecha_modelo, municipio, tipo, fecha_outlier, " \
                                "indicador, transaccion) VALUES ( %s);" % \
                                list(outliers[['f_model', 'mun', 'type', 'time', 'coefhat', 'trans']].iloc[n].values)
            # print("query 1 ok")
            file = open(plot_path + 'Outliers_check.txt', 'a')
            file.write(outliers.to_string(index=False, justify='left'))
            file.write("\n")
            file.close()
            # print("tabla 1 ok")
        # # Stores the dates of structural changes
        # print(mun, '\t', len(ans[4]) > 0)
        if len(ans[4]) > 0:
            # cambios = pd.DataFrame(pandas2ri.ri2py(ans[4]), columns=["time"])
            cambios = pandas2ri.ri2py(ans[4])
            cambios.rename(columns={cambios.columns[0]: 'time'}, inplace=True)
            cambios['f_model'] = time.strftime('%Y%m%d')
            cambios['mun'] = mun
            cambios['type'] = 'BP'
            cambios['coefhat'] = 1
            cambios['trans'] = trans
            for n in range(len(cambios)):
                query_append += "\nINSERT INTO DASHBOARD_outliers (fecha_modelo, municipio, tipo, fecha_outlier, " \
                                "indicador, transaccion) VALUES ( %s);" % \
                                list(cambios[['f_model', 'mun', 'type', 'time', 'coefhat', 'trans']].iloc[n].values)
            # print("query 2 ok")
            file = open(plot_path + 'Outliers_check.txt', 'a')
            file.write(cambios.to_string(index=False, justify='left'))
            file.write("\n")
            file.close()
            # print("tabla 2 ok")
        # # Stores the 6-month forecast for 95% confidence
        # print(mun, '\t', len(ans[6]) > 0)
        if len(ans[6]) > 0:
            outl = pd.DataFrame(pandas2ri.ri2py(ans[6]), columns=["mean", "lower", "upper"])
            outl['f_model'] = time.strftime('%Y%m%d')
            outl['mun'] = mun
            outl['fecha'] = list(map(lambda x: serie.index[-1].year*100 + serie.index[-1].month + x + 1, range(6)))
            outl['trans'] = trans
            for n in range(len(outl)):
                query_append += "\nINSERT INTO DASHBOARD_forecast (fecha_modelo, municipio, fecha_proyeccion, " \
                                "valor, upper, lower, transaccion) VALUES ( %s);" % \
                                list(outl[['f_model', 'mun', 'fecha', 'mean', 'upper', 'lower',
                                           'trans']].iloc[n].values)
            # print("query 3 ok")
            file = open(plot_path + 'Forecast_check.txt', 'a')
            file.write(outl.to_string(index=False, justify='left'))
            file.write("\n")
            file.close()
            # print("tabla 3 ok")
        # Writes results in the DB
        if len(query_append) > 0:
            # query_append = re.sub('\(\[|\[|\]|, dtype=object\)|array| 00:00:00|\n       ', '', query_append)
            query_append = re.sub('\(\[|\[|\]|, dtype=object\)|array| 00:00:00|\n +', '', query_append)
            print(query_append)
            con = pypyodbc.connect(cs_sql)
            cursor = con.cursor()
            cursor.execute(query_append)
            con.commit()
            con.close()
        # print(query_append)
    print(os.getpid(), " finalizado ok ", mun)
    return


"""
Procsa el nombre y ruta de cada municipio para crear el objeto html que lo contendra
returns: str que contiene los elementos del menu desplegable y el vinculo
"""


def f_htmltag(name: str, path: str) -> str:
    """
    Receives tha URL and the name of the object it points to in dorder to create the html link tag

    :param name: str contains the name/title of the (html) plot.
    :param path: str contains the path to the (html) plot.
    :return: str. HTML tag with the link and title. This tag is part of a dropdown menu
    """
    html_tag = '\t<option value="' + path + '">' + name + '</option>\n'
    return html_tag



def f_htmltag2(path: str) -> str:
    """
    Receives the path to the default frame selected.

    :param path: str contains the URL
    :return: str. HTML iframe tag.
    """
    html_tag = '\t<iframe style="background-color:rgba(255,255,255,1);postition:relative;top:0px;width:80%" ' \
               'frameborder="0" allowfullscreen src="' + path + '" width =2000 height= 1000></iframe>'
    return html_tag


"""
Recupera la información de las bases de datos y guarda un archivo local
returns: None. Guarda un csv con los datos leidos
"""


def f_read(infile: str, outfile: str) -> None:
    """
    Recibe una ruta de lectura que lleva a un query en cs_params y una ruta de salida para guardar los resultados.
    La funcion ejecuta el query y guarda los resultados en un csv.

    :param infile: str que contiene la ruta de letura del query.
    :param outfile: str que contiene la ruta donde se guardara el rsultado del query.
    :return: None. Los resultados se guardan en un archivo.
    """
    # Carga el query de recuperacion de datos
    sql_command = open(infile, 'r', encoding='latin1').read()
    sql_command = re.sub('\ufeff|\n|\t', ' ', sql_command)
    # Se conecta a la base para hacer los cruces
    con = pypyodbc.connect(cs_sql)
    cursor = con.cursor()
    cursor.execute(sql_command)
    cols = cursor.description
    cols = pd.DataFrame(cols)
    out = cursor.fetchall()
    out = pd.DataFrame(out)
    out.columns = cols[0].tolist()
    con.commit()
    con.close()
    out = out.sort_values(['fecha'])
    out = out[out['fecha'] > 200900]
    data0 = out.copy()
    data0.to_csv(outfile + '.csv', sep=';', index=False, encoding='latin1')
    return


"""
Grafica estandar linea de tiempo
returns: None
"""


def plot_raw(df: pd.DataFrame, col_names: list, path: str, df_name: str) -> None:
    """
    Grafica las series de tiempo de las columnas presentes en _df_.

    :param df: DataFrame con  las series de tiempo a graficar.
    :param col_names: lista de str con los nombres de las columnas de _df_.
    :param path: str que contiene la ruta donde se guardara el archivo con las graficas.
    :param df_name: str con el nombre de la transaccion.
    :return: None. Los resultados se guardan en un archivo.
    """
    if len(df.columns) == len(col_names):
        pp = PdfPages(path + 'Resumen_' + df_name + '.pdf')
        metadata = pp.infodict()
        metadata['Title'] = 'Resumen de transacciones en efectivo por municipio'
        metadata['Author'] = 'SAE, UIAF'
        metadata['Subject'] = 'Resumen gráfico de las transacciones de ' + df_name
        # Crea la hoja de portada
        plt.figure(0, figsize=(12, 6))
        plt.text(.5, .6, 'Análisis de las series de tiempo\n' + df_name, fontsize=26, va='top', ha='center', wrap=True)
        plt.text(.5, .45, '\n\nUIAF', fontsize=20, va='top', ha='center', wrap=True)
        plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
        plt.savefig(pp, format='pdf')
        plt.close()
        for col in df.columns:
            plt.figure(0, figsize=(12, 6))
            plt.plot(df[col], color='b', label="Serie original")
            valmax = np.max(df[col])
            plt.axes().format_xdata = mdates.DateFormatter('%Y-%m')
            plt.xlabel('Fecha')
            plt.suptitle('Linea de tiempo de las transacciones', fontsize=18)
            plt.title('Valor de las transacciones en pesos constantes a hoy - ' + col_names[col])
            # plt.plot((np.nan, np.nan), (np.nan, np.nan), 'r--', label='Elecciones Presidenciales', linewidth=3)
            # plt.plot((np.nan, np.nan), (np.nan, np.nan), 'g--', label='Elecciones Regionales', linewidth=3)
            years = mdates.YearLocator()
            t_exp = np.log10(valmax) // 1
            if t_exp >= 12:
                plt.axes().yaxis.set_major_formatter(formatobillones)
                plt.ylabel('Billones')
            elif t_exp >= 9:
                plt.axes().yaxis.set_major_formatter(formatomilmillones)
                plt.ylabel('Miles de Millones')
            else:
                plt.axes().yaxis.set_major_formatter(formatomillones)
                plt.ylabel('Millones')
            plt.axes().yaxis.grid(True)
            plt.yticks(rotation=0, wrap=False, ha='right')
            plt.xticks(rotation=0, wrap=False, ha='center')
            plt.figtext(0, 0, 'Fuente: UIAF.\n', fontweight='bold')
            plt.subplots_adjust(right=0.75, left=0.1)
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=False, ncol=1)
            plt.axes().xaxis.set_major_locator(years)
            # for fecha in f_nacionales:
            #     plt.plot((fecha, fecha), (0, valmax), 'r--', alpha=0.4, linewidth=1)
            # for fecha in f_locales:
            #     plt.plot((fecha, fecha), (0, valmax), 'g--', alpha=0.4, linewidth=1)
        plt.savefig(pp, format='pdf')
        plt.close()
        pp.close()


"""
Grafica linea de tiempo con outliers y pronostico
returns: Plot
"""


def plot_summary(df_raw: pd.DataFrame, df_fc: pd.DataFrame, df_outl: pd.DataFrame, col_names: list,
                 path: str, name_df: str) -> None:
    """
    Grafica las series de tiempo de las columnas presentes en _df_.

    :param df_raw: DataFrame con  las series de tiempo a graficar.
    :param df_fc: DataFrame con  las series de tiempo a graficar.
    :param df_outl: DataFrame con  las series de tiempo a graficar.
    :param col_names: lista de str con los nombres de las columnas de _df_.
    :param path: str que contiene la ruta donde se guardara el archivo con las graficas.
    :param name_df: str con el nombre de la transaccion.
    :return: None. Los resultados se guardan en un archivo.
    """
    if len(df_raw.columns) == len(col_names):
        pp = PdfPages(path + 'Resumen_' + name_df + '.pdf')
        metadata = pp.infodict()
        metadata['Title'] = 'Resumen de transacciones por municipio'
        metadata['Author'] = 'SAE, UIAF'
        metadata['Subject'] = 'Resumen gráfico de las transacciones de ' + name_df
        # Crea la hoja de portada
        plt.figure(0, figsize=(12, 6))
        plt.text(.5, .6, 'Análisis de las series de tiempo', fontsize=26, va='top', ha='center', wrap=True)
        plt.text(.5, .5, '\n\nUIAF', fontsize=20, va='top', ha='center', wrap=True)
        plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
        plt.savefig(pp, format='pdf')
        plt.close()
        # Corre el ciclo para agregar todos los municipios
        x_pos = range(0, len(df_raw) + 1)
        x_tags = list(map(lambda x: re.sub(r' 00:00:00', '', str(x)), df_raw.index))
        for col in df_raw.columns:
            plt.figure(0, figsize=(12, 6))
            plt.plot(x_pos, df_raw[col], color='g', label="Serie original")
            plt.plot(x_pos, df_raw[col].rolling(window=2, min_periods=1).mean(), color='g', alpha=0.5,
                     label="Media 12 meses")
            plt.bar(x_pos, df_outl, label="Outliers")  # ticklabels = list(map(lambda x: str(x), df_raw.index))
            plt.xticks(x_pos, x_tags)
            valmax = np.max(df_raw[col])
            plt.axes().format_xdata = mdates.DateFormatter('%Y-%m')
            plt.xlabel('Fecha')
            plt.suptitle('Linea de tiempo de las transacciones', fontsize=18)
            plt.title('Valor de las transacciones en pesos constantes a hoy - ' + col_names[col])
            years = mdates.YearLocator()
            t_exp = np.log10(valmax) // 1
            if t_exp >= 12:
                plt.axes().yaxis.set_major_formatter(formatobillones)
                plt.ylabel('Billones')
            elif t_exp >= 9:
                plt.axes().yaxis.set_major_formatter(formatomilmillones)
                plt.ylabel('Miles de Millones')
            else:
                plt.axes().yaxis.set_major_formatter(formatomillones)
                plt.ylabel('Millones')
            plt.axes().yaxis.grid(True)
            plt.yticks(rotation=0, wrap=False, ha='right')
            plt.xticks(rotation=0, wrap=False, ha='center')
            plt.figtext(0, 0, 'Fuente: UIAF.\n', fontweight='bold')
            plt.subplots_adjust(right=0.75, left=0.1)
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=False, ncol=1)
            plt.axes().xaxis.set_major_locator(years)
        plt.savefig(pp, format='pdf')
        plt.close()
        pp.close()
    return


"""
Codifica los outliers para poder guardarlos todos en una sola tabla
returns: tres listas.
"""


def d_outl(x: int, y: str) -> int:
    ans = str(x)+y
    switcher = {
        '1AO': 1,
        '-1AO': 2,
        '1LS': 3,
        '-1LS': 4,
        '1BP': 5
    }
    return switcher.get(ans, 6)


# """
# Calcula los errores estandarizados para la prueba de CUSUM
# returns: tres listas. Errores, y margenes de confiacs_paramsa superior e inferior
# """
#
#
# def f_cusum(ts: pd.Series, pdq: list, pdqs: list) -> list:
#     errores = list()
#     k = sum(pdq) + sum(pdqs[3] * pdqs[:3]) + 30
#     n = ts.count()
#     # obs = n - k
#     modelo = sarimax.SARIMAX(ts, order=pdq, seasonal_order=pdqs, trend='ct').fit(disp=False, maxiter=500)
#     params0 = pd.DataFrame(index=modelo.params.index)
#     params0['coef'] = list(modelo.params.values)
#     params0 = pd.concat([params0, modelo.conf_int(alpha=0.05)], axis=1)
#     params = pd.DataFrame(index=modelo.params.index)
#     # Diagnostics plot
#     modelo.plot_diagnostics(figsize=(12, 6))
#     plt.suptitle('Diagnostics', fontsize=20)
#     plt.savefig('DashboardEfectivo/Plots/Estimation_Diagnostics.pdf', format='pdf')
#     plt.close()
#     for t in np.arange(k, n):
#         modelo = sarimax.SARIMAX(ts[:t], order=pdq, seasonal_order=pdqs, trend='ct').fit(disp=False, maxiter=200)
#         params[t] = list(modelo.params.values)
#         sigma_e = np.mean(np.power(modelo.resid, 2))  # cual es el r
#         sigma_fc = sigma_e  ## debe ser Var(yt) + Var(et)
#         e = (ts[t] - modelo.forecast(1)[0]) / sigma_fc
#         errores.append(np.divide(e, np.sqrt(1)))
#     params = params.T
#     # w_t = np.divide(e, np.sqrt(rt))
#     upper_bound = []
#     lower_bound = np.multiply(upper_bound, -1)
#     # print(ts)
#     return [params0, params, errores, upper_bound, lower_bound]
