#ifndef TRATAMIENTOIMAGENES_H
#define TRATAMIENTOIMAGENES_H

#include <vector>
#include <cmath>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "BufferImage.h"

// Variables globales para direcciones (las hacemos constates para que sea seguro incluirlo en un .h)
static int dx[] = {0, 1, 1, 1, 0, -1, -1, -1, 0};
static int dy[] = {-1, -1, 0, 1, 1, 1, 0, -1, -1};

class TratamientoImagenes
{
public:
    static BufferImage convertirAGrisesPromedio(const cv::Mat &imagenEntrada)
    {
        BufferImage imagenSalida(imagenEntrada.cols, imagenEntrada.rows);

#pragma omp parallel for collapse(2)
        for (int x = 0; x < imagenEntrada.cols; ++x)
        {
            for (int y = 0; y < imagenEntrada.rows; ++y)
            {
                // OpenCV carga en formato BGR (Blue-Green-Red) por defecto
                cv::Vec3b pixel = imagenEntrada.at<cv::Vec3b>(y, x);
                int b = pixel[0];
                int g = pixel[1];
                int r = pixel[2];
                int nivelGris = (r + g + b) / 3;
                imagenSalida.setPixel(x, y, nivelGris);
            }
        }
        return imagenSalida;
    }

    static BufferImage ecualizarHistograma(const BufferImage &imagenEntrada)
    {
        int width = imagenEntrada.getWidth();
        int height = imagenEntrada.getHeight();
        int totalPixeles = width * height;
        std::vector<int> histograma(256, 0);
        BufferImage imagenEcualizada(width, height);
        // LUT: Si la imagen es muy grisaces, esto la distribuye para usar negros puros y blancos puros
        std::vector<float> LUT(256);
        int suma = 0;

#pragma omp parallel
        {
#pragma omp for collapse(2) // Medir tiempo con y sin paralelizacion
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    int pixel = imagenEntrada.getPixel(x, y);
                    if (pixel >= 0 && pixel < 256)
#pragma omp atomic
                        histograma[pixel]++;
                }
            }

#pragma omp single
        {
            for (int i = 0; i < 256; i++)
            {
                suma += histograma[i];
                LUT[i] = (float)suma * 255 / totalPixeles;
            }
        }

#pragma omp for collapse(2)
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    int pixel = imagenEntrada.getPixel(x, y);
                    int nuevoPixel = (int)LUT[pixel];
                    imagenEcualizada.setPixel(x, y, nuevoPixel);
                }
            }
        }
        return imagenEcualizada;
    }

    static void calcularEstadisticas(BufferImage &imagen)
    {
        int max_val = -1;
        int min_val = 300;
        long long suma = 0;
        int total = imagen.getWidth() * imagen.getHeight();

#pragma omp parallel for collapse(2) reduction(max : max_val) reduction(min : min_val) reduction(+ : suma) // Probar con y sin paralelizacion
        for (int x = 0; x < imagen.getWidth(); x++)
        {
            for (int y = 0; y < imagen.getHeight(); y++)
            {
                int pixel = imagen.getPixel(x, y);
                if (pixel > max_val)
                    // Si no funciona los reduction poner "#pragma omp atomic" aqui y quitar los reduction
                    max_val = pixel;
                if (pixel < min_val)
                    // Si no funciona los reduction poner "#pragma omp atomic" aqui y quitar los reduction
                    min_val = pixel;
                suma += pixel;
            }
        }
        imagen.setEstadisticas(max_val, min_val, (float)suma / total);
    }

    static BufferImage binarizarImagen(const BufferImage &imagen)
    {
        int width = imagen.getWidth(), height = imagen.getHeight();
        float umbral = imagen.getMedia() * 1.4f; // Umbral ajustado al 90% de la media para obtener un resultado más equilibrado
        BufferImage salida(width, height);

#pragma omp parallel for collapse(2)
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                int pixel = imagen.getPixel(x, y);
                salida.setPixel(x, y, (pixel > umbral) ? 255 : 0);
            }
        }
        return salida;
    }

    static BufferImage filtroBinario1(const BufferImage &imagen)
    {
        int width = imagen.getWidth(), height = imagen.getHeight();
        BufferImage salida(width, height);

#pragma omp parallel for collapse(2)
        for (int x = 1; x < width - 1; x++)
        {
            for (int y = 1; y < height - 1; y++)
            {
                int p = (imagen.getPixel(x, y) == 255) ? 1 : 0;
                int b = (imagen.getPixel(x, y - 1) == 255) ? 1 : 0;
                int g = (imagen.getPixel(x - 1, y) == 255) ? 1 : 0;
                int d = (imagen.getPixel(x + 1, y) == 255) ? 1 : 0;
                int e = (imagen.getPixel(x, y + 1) == 255) ? 1 : 0;

                // Buscamos rellenar huecos o eliminar pixeles que no tienen suficientes vecinos conectados
                int nuevoPixel = p | (b & g & (d | e)) | (d & e & (b | g));
                salida.setPixel(x, y, (nuevoPixel != 0) ? 255 : 0);
            }
        }
        return salida;
    }

    static BufferImage filtroBinario2(const BufferImage &imagen)
    {
        int width = imagen.getWidth(), height = imagen.getHeight();
        BufferImage salida(width, height);

#pragma omp parallel for collapse(2)
        for (int x = 1; x < width - 1; x++)
        {
            for (int y = 1; y < height - 1; y++)
            {
                int a = (imagen.getPixel(x - 1, y - 1) == 255) ? 1 : 0;
                int b = (imagen.getPixel(x, y - 1) == 255) ? 1 : 0;
                int c = (imagen.getPixel(x + 1, y - 1) == 255) ? 1 : 0;
                int d = (imagen.getPixel(x - 1, y) == 255) ? 1 : 0;
                int e = (imagen.getPixel(x + 1, y) == 255) ? 1 : 0;
                int f = (imagen.getPixel(x - 1, y + 1) == 255) ? 1 : 0;
                int g = (imagen.getPixel(x, y + 1) == 255) ? 1 : 0;
                int h = (imagen.getPixel(x + 1, y + 1) == 255) ? 1 : 0;
                int p = (imagen.getPixel(x, y) == 255) ? 1 : 0;

                int term1 = (a | b | d) & (e | g | h);
                int term2 = (b | c | e) & (d | f | g);
                int nuevoPixel = p & (term1 | term2);
                salida.setPixel(x, y, (nuevoPixel != 0) ? 255 : 0);
            }
        }
        return salida;
    }

    static BufferImage filtroSobel(const BufferImage &imagen)
    {
        int width = imagen.getWidth(), height = imagen.getHeight();
        BufferImage salida(width, height);
#pragma omp parallel for collapse(2)
        for (int x = 1; x < width - 1; x++)
        {
            for (int y = 1; y < height - 1; y++)
            {
                int Gx = (-1 * imagen.getPixel(x - 1, y - 1)) + (0 * imagen.getPixel(x, y - 1)) + (1 * imagen.getPixel(x + 1, y - 1)) +
                         (-2 * imagen.getPixel(x - 1, y)) + (0 * imagen.getPixel(x, y)) + (2 * imagen.getPixel(x + 1, y)) +
                         (-1 * imagen.getPixel(x - 1, y + 1)) + (0 * imagen.getPixel(x, y + 1)) + (1 * imagen.getPixel(x + 1, y + 1));

                int Gy = (1 * imagen.getPixel(x - 1, y - 1)) + (2 * imagen.getPixel(x, y - 1)) + (1 * imagen.getPixel(x + 1, y - 1)) +
                         (0 * imagen.getPixel(x - 1, y)) + (0 * imagen.getPixel(x, y)) + (0 * imagen.getPixel(x + 1, y)) +
                         (-1 * imagen.getPixel(x - 1, y + 1)) + (-2 * imagen.getPixel(x, y + 1)) + (-1 * imagen.getPixel(x + 1, y + 1));

                int magnitud = std::min(255, static_cast<int>(std::sqrt(Gx * Gx + Gy * Gy)));
                salida.setPixel(x, y, magnitud);
            }
        }
        return salida;
    }
};

#endif