# Fem2d_MPI

El presente repositorio contiene todo lo relacionado con el desarrollo del proyecto final para las materias de Introducción A La Computación Paralela y Computación De Alto Rendimiento Y Científica del primer semestre de 2022 de la Universidad Industrial de Santander.

## 1. Identificando las oportunidades de paralelización

Como parte del proceso de paralelizar el código dado, lo primero a realizar está en identificar que secciones del código son las que más tiempo consumen. Para esto, usando el perfilador `gprof`, se ejecutó el código serial con `nx = 70` y `ny = 70`. Esto arrojó los siguientes resultados:

```
Flat profile:

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total
 time   seconds   seconds    calls   s/call   s/call  name
 99.38    491.63   491.63        1   491.63   491.63  r8ge_fs_new(int, double*, double*)
  0.62    494.71     3.08                             main
  0.00    494.71     0.00   180000     0.00     0.00  std::setw(int)
  0.00    494.71     0.00    23096     0.00     0.00  exact(double, double, double*, double*, double*)
  0.00    494.71     0.00        2     0.00     0.00  timestamp()
  0.00    494.71     0.00        1     0.00     0.00  __static_initialization_and_destruction_0(int, int)
```

De esta manera, se identifico que, en realidad, la tarea que más tiempo consumió era la función `r8ge_fs_new` la cual se encarga de resolver el sistema de ecuaciones. Siendo así se concentraron los esfuerzos en paralelizar esta sección del código.

## Desarrollo Por

-   Didier Fernando Vallejo Sanabria - [yiye77](https://github.com/yiye77)
-   Sebastian Camilo Viancha Bautista - [SCBViancha](https://github.com/SCBViancha)
-   Yelitza Juliana Villamizar Guerrero - [Ywashere](https://github.com/Ywashere)
-   Daniel David Delgado Cervantes - [mrdahaniel](https://github.com/mrdahaniel)
-   Laura Daniela Medina Paipilla - [lauradanielamedina](https://github.com/lauradanielamedina)
-   Andrea Juliana Urrego Paredes - [Juliana18p](https://github.com/Juliana18p)
