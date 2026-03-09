#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp> // LIBRERÍA NECESARIA
#include <omp.h>
#include <filesystem> // Necesario para recorrer carpetas

#include "FingerPrintImage.h"
#include "Minutia.h"
#include "ZhangSuen.h"
#include "TratamientoImagenes.h"
#include "GestorArchivos.h"

namespace fs = std::filesystem;
int nueroHilos = 4;

int main()
{
    // Carga de imagenes (rutas)
    std::string carpetaEntrada = "C:\\Users\\gonza\\Desktop\\Universidad\\TFG\\CodigoSecuencial\\BDHuellas\\";
    std::string carpetaModificadas = "C:\\Users\\gonza\\Desktop\\Universidad\\TFG\\CodigoSecuencial\\Huellas_modificadas\\";
    std::string rutaBaseDatos = "C:\\Users\\gonza\\Desktop\\Universidad\\TFG\\CodigoSecuencial\\BaseDatosMinucias.txt";
    try
    {
        std::vector<std::string> listaArchivos;

        // Recorremos la carpeta y metemos las imagenes en una lista
        for (const auto &entrada : fs::directory_iterator(carpetaEntrada))
            if (entrada.path().extension() == ".jpg")
                listaArchivos.push_back(entrada.path().string());

        int totalHuellas = static_cast<int>(listaArchivos.size());
        std::cout << "Se han encontrado " << totalHuellas << " imagenes. Iniciando proceso..." << std::endl;

        omp_set_num_threads(numeroHilos);

        double inicio = omp_get_wtime(); // Inicio para el tiempo de procesamiento

        // Conversiones y algoritmos
        for (int i = 0; i < totalHuellas; i++)
        {
            std::string rutaActual = listaArchivos[i];
            cv::Mat imagenActual = cv::imread(rutaActual);
            if (imagenActual.empty())
                continue;

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
            GestorArchivos::guardarMinucias(rutaBaseDatos, minucias);

            // Generar salida visual (BufferedImage -> cv::Mat)
            cv::Mat salidaMinucias = TratamientoImagenes::convertirAGrisesRGBMinucias(imagenZhangSuen, minucias);

            std::string nombreBase = fs::path(rutaActual).stem().string(); // Obtiene el nombre del archivo sin extension
            std::string rutaSalida = carpetaModificadas + nombreBase + "_esqueleto.jpg";

            // Guardar la imagen (ImageIO.write -> cv::imwrite)
            cv::imwrite(rutaSalida, salidaMinucias);

            // Feedback visual para cada 100 huellas // TODO: Quitar en un futuro
            if ((i + 1) % 100 == 0)
            {
                std::cout << "Progreso: " << (i + 1) << "/" << totalHuellas << std::endl;
            }
        }

        float fin = omp_get_wtime(); // Fin para el tiempo de procesamiento
        std::cout << "\nProceso finalizado." << std::endl;

        // Leemos el achivo y lo cargamos en una matriz en la RAM
        std::vector<std::vector<Minutia>> matrizRAM = GestorArchivos::cargarBaseDatos(rutaBaseDatos);

        std::cout << "ESTADO DE LA BASE DE DATOS:" << std::endl;
        std::cout << "-----------------------------" << std::endl;
        std::cout << "Huellas registradas en memoria: " << matrizRAM.size() << std::endl;
        if (!matrizRAM.empty())
        {
            std::cout << "La ultima huella cargada tiene " << matrizRAM.back().size() << " minucias." << std::endl;
        }
        std::cout << "-----------------------------" << std::endl;

        std::cout << "Imagenes creadas y guardadas correctamente." << std::endl;
        std::cout << "Tiempo de procesamiento: " << (fin - inicio) << " segundos" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Excepcion: " << e.what() << std::endl;
    }

    return 0;
}
