#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp> // LIBRERÍA NECESARIA
#include <omp.h>
#include <filesystem> // Necesario para recorrer carpetas

#include "src/FingerPrintImage.h"
#include "src/Minutia.h"
#include "src/ZhangSuen.h"
#include "src/TratamientoImagenes.h"
#include "src/GestorArchivos.h"

namespace fs = std::filesystem;
int numeroHilos = 4;

int main()
{
    // Carga de imagenes (rutas)
    cv::Mat imagenActual = cv::imread("./assets/Huella.jpg");
    try
    {

        omp_set_num_threads(numeroHilos);

        double inicio = omp_get_wtime(); // Inicio para el tiempo de procesamiento

        // Pasa de color a gris
        FingerPrintImage imagenGrisesA = TratamientoImagenes::convertirAGrisesPromedio(imagenActual);
        // Mejora el contraste
        FingerPrintImage imagenEcualizada = TratamientoImagenes::ecualizarHistograma(imagenGrisesA);
        // Calcula el promedio de luz para usarlo como umbral
        TratamientoImagenes::calcularEstadisticas(imagenEcualizada);
        // Convierte todo a blanco (255) o negro (0)
        FingerPrintImage imagenByN = TratamientoImagenes::binarizarImagen(imagenEcualizada);
        // Limpian ruido
        FingerPrintImage imagenFiltrada1 = TratamientoImagenes::filtroBinario1(imagenByN);
        FingerPrintImage imagenFiltrada2 = TratamientoImagenes::filtroBinario2(imagenFiltrada1);

        // Algoritmo de Zhang-Suen (Adelgaza las lineas)
        FingerPrintImage imagenZhangSuen = ZhangSuen::thinning(imagenFiltrada2);

        // Minucias (encuentra donde termina o se dividen las lineas)
        std::vector<Minutia> minucias = TratamientoImagenes::detectarMinucias(imagenZhangSuen);
        // Calcula la orientacion de cada minucia
        TratamientoImagenes::calcularAngulos(imagenZhangSuen, minucias);

        // Escribimos las minucias en el archivo de texto
        // GestorArchivos::guardarMinucias(rutaBaseDatos, minucias); //TODO: cambiar la ruta destino

        // Generar salida visual (BufferedImage -> cv::Mat)
        cv::Mat salidaMinucias = TratamientoImagenes::convertirAGrisesRGBMinucias(imagenZhangSuen, minucias);

        // Guardar la imagen (ImageIO.write -> cv::imwrite)
        // cv::imwrite(rutaSalida, salidaMinucias); //TODO: :cambiar la ruta destino

        float fin = omp_get_wtime(); // Fin para el tiempo de procesamiento
        std::cout << "\nProceso finalizado." << std::endl;

        std::cout << "Imagenes creadas y guardadas correctamente." << std::endl;
        std::cout << "Tiempo de procesamiento: " << (fin - inicio) << " segundos" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Excepcion: " << e.what() << std::endl;
    }

    return 0;
}
