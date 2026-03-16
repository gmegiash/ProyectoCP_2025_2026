#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp> // LIBRERÍA NECESARIA
#include <omp.h>
#include <filesystem> // Necesario para recorrer carpetas

#include "src/FingerPrintImage.h"
#include "src/ZhangSuen.h"
#include "src/TratamientoImagenes.h"

int numeroHilos = 4;

void imprimirImagen(FingerPrintImage &imagen)
{
    cv::Mat imagenSalida(imagen.getHeight(), imagen.getWidth(), CV_8UC3);
#pragma omp parallel
    {
#pragma omp for collapse(2)
        // Copiar la huella (fondo blanco/negro) a la imagen a color
        for (int x = 0; x < imagen.getWidth(); ++x)
        {
            for (int y = 0; y < imagen.getHeight(); ++y)
            {
                int nivelGris = imagen.getPixel(x, y);
                cv::Vec3b color;
                // Si es 255 (fondo) -> blanco, si es 0 (huella) -> negro
                if (nivelGris == 255)
                    color = cv::Vec3b(255, 255, 255);
                else
                    color = cv::Vec3b(0, 0, 0);

                imagenSalida.at<cv::Vec3b>(y, x) = nivelGris;
            }
        }
    }
    cv::imwrite("./output/HuellaProcesada.jpg", imagenSalida);
}

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

        FingerPrintImage imagen = imagenEcualizada;
        imprimirImagen(imagen);

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
