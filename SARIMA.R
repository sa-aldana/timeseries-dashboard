est_sarima <- function(serie0, nombre, startYear=2009, startMonth=1){

  ff = file("C:/Users/saldana/PycharmProjects/untitled/DashboardEfectivo/Plots/R/log_R.txt", open="wt")
  sink(ff, append = F, split = F, type = "message")
  #### Imports  ####
  try(invisible(capture.output(suppressWarnings(library(forecast,    quietly = T, warn.conflicts = F, verbose = F)))))
  try(invisible(capture.output(suppressWarnings(library(tseries,     quietly = T, warn.conflicts = F, verbose = F)))))
  try(invisible(capture.output(suppressWarnings(library(tsoutliers,  quietly = T, warn.conflicts = F, verbose = F)))))
  try(invisible(capture.output(suppressWarnings(library(strucchange, quietly = T, warn.conflicts = F, verbose = F)))))
  # , lib.loc = "C:/Users/saldana/Documents/R/win-library/3.4/"
  # capture.output(suppressWarnings(library(foreach, quietly = T, warn.conflicts = F, verbose = F)))
  sink(type = "message")

  #### Definicion del entorno ####
  serie = ts(serie0, frequency = 12, start=c(startYear, startMonth))
  serie = log(serie)

  wd = getwd()
  setwd("C:/Users/saldana/PycharmProjects/untitled/DashboardEfectivo/Plots/R")
  par(mfrow=c(1,1))
  pdf(file=paste0("Plots_", nombre,".pdf"), onefile = T)

  # # Solo AR
  # l_orden = apply(expand.grid(c(1,2,3,4,5,6), c(1), c(0)), 1, FUN = function(c) paste(c, collapse = " "))
  # l_estacional = apply(expand.grid(c(0,1,2), c(0), c(0)), 1, FUN = function(c) paste(c, collapse = " "))

  # ARMA
  l_orden = apply(expand.grid(c(0,1,2,3,4), c(0), c(0,1,2,3,4)), 1, FUN = function(c) paste(c, collapse = " "))
  l_estacional = apply(expand.grid(c(0,1,2), c(0), c(0,1,2)), 1, FUN = function(c) paste(c, collapse = " "))

  #### Definicion de funciones ####

  # Funcion de estimacion del modelo
  ic_modelo  = function(order0, season, series)
  {
    tryCatch({
      modelo = Arima(y=series, order= strtoi(as.list(strsplit(order0, " "))[[1]], base=10),
                             seasonal= list(order = strtoi(as.list(strsplit(season, " "))[[1]], base=10), period=12),
                             include.mean = TRUE, transform.pars = TRUE, method = "CSS-ML")
      df = data.frame(modelo$aicc, row.names = "ic")
      colnames(df) <- paste(order0, season, sep = "_")
      df
      }, error=function(e) NA)
  }

  # Funcion de comparacion de fechas
  compareDates = function(x, largo)
  {
    l_fecha = (0:(largo-1) + startMonth-1)/12 + startYear
    l_fecha = floor(l_fecha) *100 + (l_fecha %% 1) *12 +1
    val = strtoi(as.list(strsplit(sub(")", "", x, fixed = T), "(", fixed = T))[[1]], base=10)
    val = val[1]*100+val[2]
    as.numeric(l_fecha == val)
  }
  # Funcion de ajuste de formato de fechas
  fix_format = function(x)
  {
    ans = strsplit(x,"(", fixed=TRUE)[[1]]
    ans = strtoi(paste0(ans[[1]], substr(paste0("0", ans[[2]]), start=nchar(ans[[2]])-1, stop=nchar(ans[[2]]))))+0.0
    ans
  }

  ##### ----- Funcion de deteccion de outliers ----- #####
  tryCatch(
  {
    #### Estimacion del modelo adecuado ####
    # regresa una lista anidada de estacion * ordenes
    l_ic = lapply(l_estacional, function(y) {lapply(l_orden, function(x) {ic_modelo(order0 = x, season = y, series = serie)})})
    # obtiene el valor del minimo estacion * orden
    m_ic = row.names.data.frame(data.frame(which.min(unlist(l_ic))))

    # Ajuste del modelo
    c_orden = as.list(strsplit(m_ic, "_"))[[1]][[1]]
    c_orden = strtoi(as.list(strsplit(c_orden, " "))[[1]], base=10)
    c_estacion = as.list(strsplit(m_ic, "_"))[[1]][[2]]
    c_estacion = strtoi(as.list(strsplit(c_estacion, " "))[[1]], base=10)

    #### Estima el modelo sarima que reciba el autofit ####
    modelo = Arima(serie, order= c_orden, seasonal= list(order = c_estacion, period=12), include.mean = TRUE, transform.pars = TRUE, method = "CSS-ML")
    resids = modelo$residuals
    # print(modelo)
    fc = forecast(modelo, h=6, level=95)
    plot(fc)
    title(ylab = "Log(COP)", xlab= "Fecha")
    # Ajusta la tabla: valores relativos
    # media = mean(serie[(length(serie)-11):length(serie)])
    # desv = sd(serie)
    fc_table = cbind(fc$mean, fc$lower, fc$upper)
    fc_table = exp(fc_table)

    #### Deteccion de outliers ####
    outl_types= c("AO", "LS") # , "TC", "IO", "SLS"
    resultados = matrix(0, nrow = length(serie), ncol = length(outl_types))
    cof_fit = coefs2poly(modelo)
    for(i in 1:length(outl_types))
    {
      resultados[, i] = outliers.tstatistics(resids, pars = cof_fit, types = outl_types[i])[,,2]
    }

    largo = length(serie)
    largo_trim = floor(largo*0.05)
    valor_c = (sd(sort(serie)[largo_trim:(largo-largo_trim)])/sd(serie))*3
    outl = locate.outliers(resids, pars = cof_fit, types = outl_types, cval = valor_c)
    fechas = (outl$ind+startMonth-1)/12 + startYear
    outl$ind = floor(fechas) *100 + ifelse((fechas %% 1)==0, -100+12, (fechas %% 1) *12) # (fechas %% 1) *12
    outl$time = gsub(":", "", outl$ind)
    outl$coefhat = sign(outl$coefhat)

    # outl = tso(resids, types=outl_types, maxit.oloop = 10, maxit.iloop = 10, cval = 3.0,
    #            tsmethod = "auto.arima", args.tsmethod = list(order= c_orden, seasonal= list(order = c_estacion)))
    # fechas = (outl$outliers$ind+startMonth-1)/12 + startYear
    # outl$outliers$ind = floor(fechas) *100 + (fechas %% 1) *12
    # outl$outliers$time = gsub(":", "", outl$outliers$time)
    # outl$outliers$coefhat = sign(outl$outliers$coefhat)

    # rownames(resultados) <- outl_types

    #### Cambio estructural ####

    # Combina en un un dataset las variables
    l_ar = c(0:c_orden[1], if(c_estacion[1]>0) (1:c_estacion[1])*12 else c())
    l_ma = c(if(c_orden[3]>0) (1:c_orden[3]) else c(), if(c_estacion[3]>0) (1:c_estacion[3])*12 else c())
    # l_ma = l_ma[2:length(l_ma)]
    dataset = do.call(ts.union,
                      c(lapply(l_ar, FUN = function(n) lag(serie, k=-n, na.pad=T)),
                        lapply(l_ma, FUN = function(n) lag(resids, k=-n, na.pad=T))))
    l_ma = c("", l_ma)
    colnames(dataset) <- strsplit(gsub("0_","_",paste0("serie", paste(l_ar, collapse = "_ar"), paste(l_ma, collapse = "_ma"))), "_")[[1]]
    dataset = as.data.frame(dataset)
    dataset = ts(dataset, frequency = 12, start=c(2009,1))
    dataset = na.omit(dataset)
    formula = gsub("_"," + ", gsub("serie_","serie ~ ", gsub("0_","_",paste0("serie", paste(l_ar, collapse = "_ar"), paste(l_ma, collapse = "_ma")))))

    # Identificacion de cambios estructurales
    bp = breakpoints(eval(parse(text=formula)), data=dataset, h=dim(dataset)[2]+2) # , hpc = "foreach"
    bd = breakdates(bp, format.times=T)

    breaks = rowSums(sapply(t.data.frame(data.frame(bd)), FUN = function(x) {compareDates(x, length(serie))}))


    # # Pruebas de estabilidad
    par(mfrow=c(2,1))
    mosum = efp(eval(parse(text=formula)), data= dataset, h=0.1, type="Rec-MOSUM")
    plot(mosum)

    fst = Fstats(eval(parse(text=formula)), data= dataset)
    plot(fst, alpha=0.05)
    title(main = "Prueba F-sup para cambio estructural")
    par(mfrow=c(1,1))

    # max_year # modificar para que se ajuste la fecha final de las series
    plot(ts(dataset[,"serie"], frequency = 12, start=2018-length(dataset[,"serie"])/12), col="blue", type="l", ann=F)
    lines(ts(modelo$fitted, frequency = 12, start=2018-length(modelo$fitted)/12), col="red", type="l")
    lines(ts(fitted(bp, breaks = length(na.omit(bd))), frequency = 12, start=2018-length(fitted(bp, breaks = length(na.omit(bd))))/12), col="green", type="l")
    legend("topleft", legend=c("Serie original", "Modelo base", "Modelo con cambio estructural"), col=c("blue", "red","green"), lty= c(1,1), cex=0.8)
    title(main="Ajuste con cambio estructural", sub = paste0( "Hay ",length(na.omit(bd)), " punto(s) de cambio estructural"),
          ylab = "Log(X)", xlab= "Fecha")

    #### Fin de la funcion ####

    # Guarda las graficas
    dev.off()
    # print(paste0("Archivo 'Plots_", nombre,".pdf' creado exitosamente."))
    par(mfrow=c(1,1))
    setwd(wd)

    # Guardar los resultados de la funcion
    # (1) orden del modelo
    # (2) coeficientes del modelo
    # (3) outliers
    # (4) tvalue outliers
    # (5) fechas de los cambios
    # (6) posicion de los cambios
    # (7) pronostico 6 meses

    valor = list("order" = m_ic,
                 "coeficients" = cof_fit,
                 "outliers" = data.frame(cbind(outl)),
                 "serieoutliers" = resultados,
                 "breakdates" = data.frame(sapply(na.omit(bd), FUN = fix_format)),
                 "seriebreaks" = breaks,
                 "forecast" = fc_table)
    valor
  }, error= function(e){
    # Guarda las graficas
    dev.off()
    # print(paste0("Archivo 'Plots_", nombre,".pdf' creado exitosamente."))
    par(mfrow=c(1,1))
    setwd(wd)

    valor = list("order" = NULL,
                 "coeficients" = NULL,
                 "outliers" = NULL,
                 "serieoutliers" = NULL,
                 "breakdates" = NULL,
                 "seriebreaks" = NULL,
                 "forecast" = NULL)
    valor
  })
}
