#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp> // LIBRERÍA NECESARIA
#include <omp.h>
#include <filesystem> // Necesario para recorrer carpetas

#include "src/BufferImage.h"
#include "src/TratamientoImagenes.h"

namespace fs = std::filesystem;
fs::path ruta_proyecto = fs::current_path();

int main(int argc, char *argv[])
{
    int numThreads;
    if (argc != 3)
    {
        std::cout << "Uso: " << argv[0] << " <num_threads> <input_image_path>" << std::endl;
        return -1;
    }

    numThreads = std::stoi(argv[1]);
    fs::path ruta_imagen = fs::path(argv[2]);

    
    if (ruta_proyecto.filename() == "build")
        ruta_proyecto = ruta_proyecto.parent_path();

    fs::path ruta_salida = ruta_proyecto / "output";

    // Carga de imagenes (rutas)
    cv::Mat imagenActual = cv::imread(ruta_imagen.string());
    if (imagenActual.empty())
    {
        std::cerr << "ERROR: No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    try
    {
        omp_set_num_threads(numThreads);

        double inicio = omp_get_wtime(); // Inicio para el tiempo de procesamiento

        // Pasa de color a gris
        BufferImage imagenGrisesA = TratamientoImagenes::convertirAGrisesPromedio(imagenActual);
        fs::path rutaSalidaGrisesA = ruta_salida / "grises.jpg";
        imagenGrisesA.imprimirImagen(rutaSalidaGrisesA);
        // Mejora el contraste
        BufferImage imagenEcualizada = TratamientoImagenes::ecualizarHistograma(imagenGrisesA);
        fs::path rutaSalidaEcualizada = ruta_salida / "ecualizada.jpg";
        imagenEcualizada.imprimirImagen(rutaSalidaEcualizada);

        TratamientoImagenes::calcularEstadisticas(imagenEcualizada);

        // Convierte todo a blanco (255) o negro (0)
        BufferImage imagenByN = TratamientoImagenes::binarizarImagen(imagenEcualizada);
        fs::path rutaSalidaByN = ruta_salida / "byn.jpg";
        imagenByN.imprimirImagen(rutaSalidaByN);

        BufferImage imagenSobel = TratamientoImagenes::filtroSobel(imagenByN);
        fs::path rutaSalidaSobel = ruta_salida / "sobel.jpg";
        imagenSobel.imprimirImagen(rutaSalidaSobel);

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
