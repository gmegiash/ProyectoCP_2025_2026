#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp> // LIBRERÍA NECESARIA
#include <omp.h>
#include <filesystem> // Necesario para recorrer carpetas

#include "src/BufferImage.h"
#include "src/TratamientoImagenes.h"

int numeroHilos = 4;

namespace fs = std::filesystem;
fs::path ruta_proyecto = fs::current_path();

void imprimirImagen(fs::path ruta_salida, const BufferImage &imagenFinal)
{
    // Creamos una matriz de OpenCV del mismo tamaño que nuestra imagen procesada
    cv::Mat matFinal(imagenFinal.getHeight(), imagenFinal.getWidth(), CV_8UC1);
#pragma omp parallel for collapse(2)
    for (int x = 0; x < imagenFinal.getWidth(); ++x)
    {
        for (int y = 0; y < imagenFinal.getHeight(); ++y)
        {
            // Pasamos el valor del pixel directamente (0 o 255)
            matFinal.at<uchar>(y, x) = (uchar)imagenFinal.getPixel(x, y);
        }
    }

    // Mostrar la ventana con el resultado
    cv::imwrite(ruta_salida.string(), matFinal);
}

int main()
{
    if (ruta_proyecto.filename() == "build")
        ruta_proyecto = ruta_proyecto.parent_path();

    fs::path ruta_salida = ruta_proyecto / "output" / "LosDiozeProcesada.jpg";
    fs::path ruta_imagen = ruta_proyecto / "assets" / "LosDioze.jpg";

    // Carga de imagenes (rutas)
    cv::Mat imagenActual = cv::imread(ruta_imagen.string());
    if (imagenActual.empty())
    {
        std::cerr << "ERROR: No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    try
    {
        omp_set_num_threads(numeroHilos);

        double inicio = omp_get_wtime(); // Inicio para el tiempo de procesamiento

        // Pasa de color a gris
        BufferImage imagenGrisesA = TratamientoImagenes::convertirAGrisesPromedio(imagenActual);
        // Mejora el contraste
        BufferImage imagenEcualizada = TratamientoImagenes::ecualizarHistograma(imagenGrisesA);

        TratamientoImagenes::calcularEstadisticas(imagenEcualizada);

        // Convierte todo a blanco (255) o negro (0)
        BufferImage imagenByN = TratamientoImagenes::binarizarImagen(imagenEcualizada);
        // Limpian ruido
        BufferImage imagenFiltrada1 = TratamientoImagenes::filtroBinario1(imagenByN);
        BufferImage imagenFiltrada2 = TratamientoImagenes::filtroBinario2(imagenFiltrada1);

        BufferImage imagenFinal = imagenFiltrada2;

        imprimirImagen(ruta_salida, imagenFinal);

        double fin = omp_get_wtime(); // Fin para el tiempo de procesamiento
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
