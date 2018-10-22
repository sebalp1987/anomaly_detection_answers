import pandas as pd
import STRING

def clean_ine(path: str):

    df = pd.read_csv(STRING.path_ine + path, sep=',', encoding='latin1')
    var_names = pd.read_csv(STRING.path_ine + 'variable_names.csv', sep=';', encoding='latin1', names=['code', 'descripc'])
    var_names = var_names.dropna()
    var_names = var_names.set_index('code', drop=True)['descripc'].to_dict()
    df = df.rename(columns=var_names)

    final_table = df[['cpro', 'cmun', 'dist', 'secc', 'Población total', 'Total Viviendas']]

    # Hombres / Mujeres
    final_table['sexo_porcent'] = df['Hombres'] / df['Población total']

    # Edad
    final_table['edad_16'] = df['Personas de menos de 16 años'] / df['Población total']
    final_table['edad_16_64'] = df['Personas entre 16 (incluido) y 64 (incluido) años'] / df['Población total']
    final_table['edad_64'] = df['Personas con más de 64 años'] / df['Población total']


    # Nacionalidad español
    df['Personas que han nacido en España'] = df['Personas que han nacido en España'].fillna(0)
    final_table['español'] = df['Personas que han nacido en España'] / df['Población total']

    # Extranjeros
    df = df.rename(columns={'Personas que han nacido en otro estado miembro de la UE': 'UE',
                            'Personas que han nacido en un país europeo que no es miembro de la UE': 'No UE',
                            'Personas que han nacido en América Central, del Sur o Caribe': 'Latino'})

    var_nac = ['UE',
               'No UE', 'Latino',
               'Personas que han nacido en Africa',
               'Personas que han nacido en América del Norte', 'Personas que han nacido en Asia',
               'Personas que han nacido en Oceanía']

    for i in var_nac:
        df[i] = df[i].fillna(0)
        final_table[i.replace('Personas que han nacido en ', '')] = df[i] / df['Población total']

    # Estado Civil
    var_civil = ['Personas con estado civil soltero', 'Personas con estado civil casado',
                 'Personas con estado civil separado', 'Personas con estado civil divorciado',
                 'Personas con estado civil viudo']
    for i in var_civil:
        df[i] = df[i].fillna(0)
        final_table[i.replace('Personas con estado civil ', '')] = df[i] / df['Población total']


    # Litteracy
    var_estudios = ['Personas analfabetas', 'Personas sin estudios', 'Personas con estudios de primer grado',
                    'Personas con estudios de segundo grado', 'Personas con estudios de tercer grado']
    for i in var_estudios:
        df[i] = df[i].fillna(0)
        final_table[i.replace('Personas con ', '')] = df[i] / df['Población total']

    # Viviendas per cápita
    final_table['Vivienda_pc'] = df['Población total'] / df['Total Viviendas']

    # Tipo de Vivienda
    var_vivienda = ['Viviendas Principales', 'Viviendas Secundarias', 'Viviendas Vacías']
    for i in var_vivienda:
        df[i] = df[i].fillna(0)
        final_table[i] = df[i] / df['Total Viviendas']

    # Tenencia vivienda
    df = df.rename(columns={'Viviendas en propiedad, por compra, totalmente pagada':'vv_compra_pagada',
                            'Viviendas en propiedad, por compra, con pagos pendientes':'vv_compra_parcial',
                            'Viviendas en propiedad, por herencia o donación':'vv_donacion',
                            'Viviendas en alquiler':'vv_alquiler', 'Viviendas cedida gratis o a bajo precio':'vv_cedida',
                            'Viviendas en otro tipo de régimen de tenencia':'vv_otro'})

    var_tenencia = ['vv_compra_pagada', 'vv_compra_parcial', 'vv_donacion', 'vv_alquiler', 'vv_cedida', 'vv_otro']
    for i in var_tenencia:
        df[i] = df[i].fillna(0)
        final_table[i] = df[i] / df['Total Viviendas']

    # Tamaño Vivienda
    df = df.rename(columns={'Viviendas de menos de 30m2': '<30m2', 'Viviendas de más de 180 m2':'>180m2'})
    var_tamanio = ['<30m2', 'Viviendas entre 30-45 m2', 'Viviendas entre 46-60 m2',
                   'Viviendas entre 61-75 m2', 'Viviendas entre 76-90 m2', 'Viviendas entre 91-105 m2',
                   'Viviendas entre 106-120 m2', 'Viviendas entre 121-150 m2', 'Viviendas entre 151-180 m2',
                   '>180m2']

    for i in var_tamanio:
        df[i] = df[i].fillna(0)
        final_table[i.replace('Viviendas entre ', '')] = df[i] / df['Total Viviendas']

    seccion = final_table[['cpro', 'cmun', 'dist', 'secc', 'edad_16', 'No UE', 'Latino', 'Africa',
                             'Personas analfabetas', 'Personas sin estudios', 'Vivienda_pc', '151-180 m2', '>180m2']]

    # Ranking edad
    seccion['ranking_edad_16_seccion'] = seccion['edad_16'].rank(method='dense', ascending=False)
    del seccion['edad_16']

    # Ranking extranjeros Riesgo
    seccion['extranjeros_riesgo'] = seccion['No UE'] + seccion['Latino'] + seccion['Africa']
    seccion['ranking_extranjeros_riesgo'] = seccion['extranjeros_riesgo'].rank(method='dense', ascending=False)
    del seccion['No UE']
    del seccion['Latino']
    del seccion['Africa']

    # Ranking Analfabetismo
    seccion['analfabetismo'] = seccion['Personas analfabetas'] + seccion['Personas sin estudios']
    seccion['ranking_analfabetismo'] = seccion['analfabetismo'].rank(method='dense', ascending=False)
    del seccion['Personas analfabetas']
    del seccion['Personas sin estudios']

    # Ranking Densidad
    seccion['ranking_densidad'] = seccion['Vivienda_pc'].rank(method='dense', ascending=False)
    del seccion['Vivienda_pc']

    # Ranking Proxy Income
    seccion['income_by_size_property'] = seccion['151-180 m2'] + seccion['>180m2']
    seccion['ranking_income_size_property'] = seccion['income_by_size_property'].rank(method='dense', ascending=False)
    del seccion['151-180 m2']
    del seccion['>180m2']

    final_table = pd.merge(final_table, seccion, how='left', on=['cpro', 'cmun', 'dist', 'secc'])

    # Población según Provincia
    final_table.to_csv('test.csv', sep=';', index=False, encoding='latin1')
    print(final_table.head(5))


if __name__ == '__main__':
    clean_ine('C2011_ccaa01_Indicadores.csv')
