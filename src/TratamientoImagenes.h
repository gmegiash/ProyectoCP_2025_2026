#ifndef TRATAMIENTOIMAGENES_H
#define TRATAMIENTOIMAGENES_H

#include <vector>
#include <cmath>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "FingerPrintImage.h"
#include "Minutia.h"

// Variables globales para direcciones (las hacemos constates para que sea seguro incluirlo en un .h)
static int dx[] = {0, 1, 1, 1, 0, -1, -1, -1, 0};
static int dy[] = {-1, -1, 0, 1, 1, 1, 0, -1, -1};

class TratamientoImagenes
{
public:
    static FingerPrintImage convertirAGrisesPromedio(const cv::Mat &imagenEntrada)
    {
        FingerPrintImage imagenSalida(imagenEntrada.cols, imagenEntrada.rows);

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

    static FingerPrintImage ecualizarHistograma(const FingerPrintImage &imagenEntrada)
    {
        int width = imagenEntrada.getWidth();
        int height = imagenEntrada.getHeight();
        int totalPixeles = width * height;
        std::vector<int> histograma(256, 0);
        FingerPrintImage imagenEcualizada(width, height);
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

#pragma omp for reduction(+ : suma)
            for (int i = 0; i < 256; i++)
            {
                suma += histograma[i];
                LUT[i] = (float)suma * 255 / totalPixeles;
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

    static void calcularEstadisticas(FingerPrintImage &imagen)
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

    static FingerPrintImage binarizarImagen(const FingerPrintImage &imagen)
    {
        int width = imagen.getWidth(), height = imagen.getHeight();
        float umbral = imagen.getMedia();
        FingerPrintImage salida(width, height);

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

    static FingerPrintImage filtroBinario1(const FingerPrintImage &imagen)
    {
        int width = imagen.getWidth(), height = imagen.getHeight();
        FingerPrintImage salida(width, height);

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

    static FingerPrintImage filtroBinario2(const FingerPrintImage &imagen)
    {
        int width = imagen.getWidth(), height = imagen.getHeight();
        FingerPrintImage salida(width, height);

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

    // Saber hacia donde apunta la minucia en ese punto
    static double calcularAnguloRama(const FingerPrintImage &imagen, int x, int y, int startX = -1, int startY = -1)
    {
        int xi = x, yi = y, xf = x, yf = y;
        int width = imagen.getWidth();
        int height = imagen.getHeight();
        std::vector<std::vector<bool>> visitado(width, std::vector<bool>(height, false));

        int pasos = 0;
        int px = (startX == -1) ? x : startX;
        int py = (startY == -1) ? y : startY;

        if (startX != -1 && startY != -1)
        {
            xi = startX;
            yi = startY;
        }
        visitado[x][y] = true;

        while (pasos < 6)
        {
            bool encontrado = false;

            for (int dir = 0; dir < 8; dir++)
            {
                int nx = px + dx[dir];
                int ny = py + dy[dir];
                if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                {
                    continue;
                }
                if (imagen.getPixel(nx, ny) == 0 && !visitado[nx][ny])
                {
                    visitado[nx][ny] = true;
                    px = nx;
                    py = ny;
                    xf = nx;
                    yf = ny;
                    encontrado = true;
                    pasos++;
                    break;
                }
            }
            if (!encontrado)
            {
                break;
            }
        }
        int gx = xf - xi;
        int gy = yf - yi;
        // Con atan2 no falla si x=0 y devuelve un angulo entre "-pi" y "pi"
        return std::atan2((double)gy, (double)gx);
    }

    static void
    calcularAngulos(const FingerPrintImage &imagen, std::vector<Minutia> &minucias)
    {
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < minucias.size(); i++)
        {
            if (minucias[i].tipo == true)
            { // Minucia de tipo final
                minucias[i].angulo.push_back(calcularAnguloRama(imagen, minucias[i].x, minucias[i].y));
            }
            else if (minucias[i].tipo == false)
            { // Minucia de tipo bifurcacion
                std::vector<double> angulos;
                for (int dir = 0; dir < 8; dir++)
                {
                    int nx = minucias[i].x + dx[dir];
                    int ny = minucias[i].y + dy[dir];
                    if (nx >= 0 && nx < imagen.getWidth() && ny >= 0 && ny < imagen.getHeight())
                    {
                        if (imagen.getPixel(nx, ny) == 0)
                        {
                            angulos.push_back(calcularAnguloRama(imagen, minucias[i].x, minucias[i].y, nx, ny));
                        }
                    }
                }
                minucias[i].angulo = angulos;
            }
        }
    }

    static std::vector<Minutia> detectarMinucias(const FingerPrintImage &imagen)
    {
        std::vector<Minutia> minucias;
        int width = imagen.getWidth(), height = imagen.getHeight();

#pragma omp parallel for collapse(2) schedule(guided)
        for (int x = 1; x < width - 1; x++)
        {
            for (int y = 1; y < height - 1; y++)
            {
                if (imagen.getPixel(x, y) != 0)
                    continue;

                int p[9];
                p[0] = (imagen.getPixel(x, y) == 0) ? 1 : 0;
                for (int i = 0; i < 8; i++)
                {
                    int nx = x + dx[i];
                    int ny = y + dy[i];
                    p[i + 1] = (imagen.getPixel(nx, ny) == 0) ? 1 : 0;
                }

                int cn = 0;
                for (int i = 1; i <= 8; i++)
                {
                    cn += std::abs(p[i] - p[i % 8 + 1]);
                }
                cn /= 2;
                if (cn == 1)
                {
                    minucias.emplace_back(x, y, true, std::vector<double>{});
                }
                else if (cn == 3)
                {
                    minucias.emplace_back(x, y, false, std::vector<double>{});
                }
            }
        }
        return minucias;
    }

    static cv::Mat convertirAGrisesRGBMinucias(const FingerPrintImage &imagen, const std::vector<Minutia> &minucias)
    {
        // 1. Crear imagen base OpenCV RGB (CV_8UC3)
        cv::Mat imagenSalida(imagen.getHeight(), imagen.getWidth(), CV_8UC3);
        int longitudLinea = 12; // Longitud de la línea de dirección (en píxeles)

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

                    imagenSalida.at<cv::Vec3b>(y, x) = color;
                }
            }

// 2. Dibujar las Minucias y sus Ángulos
#pragma omp for
            for (int i = 0; i < minucias.size(); ++i)
            {
                const auto &m = minucias[i]; // TODO: Verificar en secuencial
                // Verificar límites por seguridad
                if (m.x < 0 || m.x >= imagen.getWidth() || m.y < 0 || m.y >= imagen.getHeight())
                    continue;

                cv::Scalar color; // Usamos Scalar para las funciones de dibujo de OpenCV

                if (m.tipo == true)
                {                                  // Minucia de tipo final
                    color = cv::Scalar(0, 0, 255); // Rojo (BGR)
                }
                else if (m.tipo == false)
                {                                  // Minucia de tipo bifurcacion
                    color = cv::Scalar(255, 0, 0); // Azul (BGR)
                }

                // A) Dibujar el punto central (Un círculo pequeño para que se vea mejor)
                // TODO: Si hay problemas, poner "#pragma omp critical"
                cv::circle(imagenSalida, cv::Point(m.x, m.y), 2, color, -1); // Radio 2, relleno (-1)

                // B) Dibujar la(s) línea(s) de dirección
                // Una minucia puede tener varios ángulos (bifurcación tiene 3, final tiene 1)
                for (double anguloRad : m.angulo)
                {
                    // Calcular el punto final usando trigonometría básica
                    // x2 = x1 + largo * cos(angulo)
                    // y2 = y1 + largo * sin(angulo)
                    int x2 = m.x + (int)(longitudLinea * std::cos(anguloRad));
                    int y2 = m.y + (int)(longitudLinea * std::sin(anguloRad));

                    cv::Point inicio(m.x, m.y);
                    cv::Point fin(x2, y2);

                    // Dibujar línea con grosor 1
                    // TODO: Si hay problemas, poner "#pragma omp critical"
                    cv::line(imagenSalida, inicio, fin, color, 1, cv::LINE_AA);
                }
            }
        }
        return imagenSalida;
    }
};

#endif