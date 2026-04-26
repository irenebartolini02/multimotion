* Encoding: UTF-8.
*______________________________________________________________________________________________________________.

* 				LABORATORIO DI ANALISI DEI DATI.
* 				T. Parisi, L. Ricolfi.

* 		     		17 aprile: INDSCAL
*______________________________________________________________________________________________________________.

*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$.

* 								INDSCAL.

*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$.


*************************************************CREAZIONE DELL'INPUT: matrici individuali impilate.

GET
  FILE='\\srvthin\HOME$\325155\work4\ind116A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind166A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind1A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind54A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind60A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind66A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind67A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind70A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind82A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind84A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind8A.sav'.

ADD FILES /FILE =* /FILE='\\srvthin\HOME$\325155\work4\ind9000A.sav'.

EXECUTE.

RENAME VARIABLES (col1 col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 = udc ds fi leg sdi ver an mar msi rad rc itv).

SAVE OUTFILES='\\srvthin\HOME$\325155\work4\Matrice impilata.sav'.



*****************************INDSCAL.

GET
  FILE='\\srvthin\HOME$\325155\work4\Matrice impilata.sav'.

ALSCAL
  VARIABLES= udc ds fi leg sdi ver an mar msi rad rc itv
  /SHAPE=SYMMETRIC
  /LEVEL=RATIO
  /CONDITION=MATRIX
  /MODEL=INDSCAL
  /CRITERIA=CONVERGE(.0000001) STRESSMIN(.0000001) ITER(1000) CUTOFF(0) DIMENS(2)
  /PLOT=default all
  /PRINT=HEADER.

GET
  FILE='\\srvthin\HOME$\325155\work3\ind9000A.sav'.


ALSCAL
  VARIABLES= udc to itv
  /SHAPE=SYMMETRIC
  /LEVEL=INTERVAL
  /CONDITION=MATRIX
  /MODEL=EUCLID
  /CRITERIA=CONVERGE(.000000001) STRESSMIN(.000000005) ITER(2000) CUTOFF(0) DIMENS(2)
  /PLOT=ALL 
 /PRINT=DATA HEADER.


GET
  FILE='\\srvthin\HOME$\325155\work3\ind8A.sav'.

RENAME VARIABLES (col1 col2 col3 col4 col5 col6 col7 col8 col9 col10 col11 col12 = udc ds fi leg sdi ver an mar msi rad rc itv).

SAVE OUTFILES='\\srvthin\HOME$\325155\work3\ind8A.sav'.

GET
  FILE='\\srvthin\HOME$\325155\work3\ind8A.sav'.


ALSCAL
  VARIABLES= udc to itv
  /SHAPE=SYMMETRIC
  /LEVEL=INTERVAL
  /CONDITION=MATRIX
  /MODEL=EUCLID
  /CRITERIA=CONVERGE(.000000001) STRESSMIN(.000000005) ITER(2000) CUTOFF(0) DIMENS(2)
  /PLOT=ALL 
 /PRINT=DATA HEADER.




**VARIARE LIVELLO DI SCALA E DIMENSIONALITA'.

GET
  FILE='\\srvthin\HOME$\325155\work4\Matrice impilata.sav'.

  ALSCAL
  VARIABLES=  A1_LP A2_LP A3_LP A4_LP A_HN B_HN C_LN F_HN G_HP H_HP J_Ne K_Ne M_LN  N_LN O_LN P_HP Q_HP U_Ne V_Ne W_HN
  /SHAPE=SYMMETRIC
  /LEVEL=INTERVAL
  /CONDITION=MATRIX
  /MODEL=INDSCAL
  /CRITERIA=CONVERGE(.0000001) STRESSMIN(.0000001) ITER(1000) CUTOFF(0) DIMENS(2)
  /PLOT=default all
  /PRINT=HEADER.
