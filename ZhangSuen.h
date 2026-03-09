#ifndef ZHANGSUEN_H
#define ZHANGSUEN_H

#include "FingerPrintImage.h"
#include <vector>

class ZhangSuen
{
public:
    static FingerPrintImage thinning(const FingerPrintImage &img)
    {
        int width = img.getWidth();
        int height = img.getHeight();
        bool change;
        FingerPrintImage salida(width, height);

// Copiar imagen original
#pragma omp parallel for collapse(2)
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++)
                salida.setPixel(x, y, img.getPixel(x, y));

        do
        {
            change = false;
            // Matriz de marcadores [width][height] iniciada en false
            std::vector<std::vector<bool>> marker(width, std::vector<bool>(height, false));

#pragma omp parallel
            {
#pragma omp for collapse(2) schedule(guided)
                // Primera subiteración
                for (int x = 1; x < width - 1; x++)
                {
                    for (int y = 1; y < height - 1; y++)
                    {
                        int p1 = (salida.getPixel(x, y) == 0) ? 1 : 0;
                        if (p1 != 1)
                            continue;

                        int neighbors[8];
                        neighbors[0] = (salida.getPixel(x, y - 1) == 0) ? 1 : 0;     // p2
                        neighbors[1] = (salida.getPixel(x + 1, y - 1) == 0) ? 1 : 0; // p3
                        neighbors[2] = (salida.getPixel(x + 1, y) == 0) ? 1 : 0;     // p4
                        neighbors[3] = (salida.getPixel(x + 1, y + 1) == 0) ? 1 : 0; // p5
                        neighbors[4] = (salida.getPixel(x, y + 1) == 0) ? 1 : 0;     // p6
                        neighbors[5] = (salida.getPixel(x - 1, y + 1) == 0) ? 1 : 0; // p7
                        neighbors[6] = (salida.getPixel(x - 1, y) == 0) ? 1 : 0;     // p8
                        neighbors[7] = (salida.getPixel(x - 1, y - 1) == 0) ? 1 : 0; // p9

                        int count = 0;
                        for (int n : neighbors)
                            count += n;
                        if (count < 2 || count > 6)
                            continue;

                        int transitions = 0;
                        for (int i = 0; i < 7; i++)
                        {
                            if (neighbors[i] == 0 && neighbors[i + 1] == 1)
                                transitions++;
                        }
                        if (neighbors[7] == 0 && neighbors[0] == 1)
                            transitions++;
                        if (transitions != 1)
                            continue;

                        if (neighbors[0] * neighbors[2] * neighbors[4] != 0)
                            continue;
                        if (neighbors[2] * neighbors[4] * neighbors[6] != 0)
                            continue;

                        marker[x][y] = true;
#pragma omp atomic // #pragma omp critical (Si funciona quedar atomic que es mas rapido)
                        change = true;
                    }
                }

#pragma omp for collapse(2) schedule(guided)
                for (int x = 1; x < width - 1; x++)
                    for (int y = 1; y < height - 1; y++)
                        if (marker[x][y])
                            salida.setPixel(x, y, 255);

// Reiniciar marcadores
#pragma omp for
                for (int i = 0; i < width; i++)
                    std::fill(marker[i].begin(), marker[i].end(), false);

// Segunda subiteración
#pragma omp for collapse(2) schedule(guided)
                for (int x = 1; x < width - 1; x++)
                {
                    for (int y = 1; y < height - 1; y++)
                    {
                        int p1 = (salida.getPixel(x, y) == 0) ? 1 : 0;
                        if (p1 != 1)
                            continue;

                        int neighbors[8];
                        neighbors[0] = (salida.getPixel(x, y - 1) == 0) ? 1 : 0;     // p2
                        neighbors[1] = (salida.getPixel(x + 1, y - 1) == 0) ? 1 : 0; // p3
                        neighbors[2] = (salida.getPixel(x + 1, y) == 0) ? 1 : 0;     // p4
                        neighbors[3] = (salida.getPixel(x + 1, y + 1) == 0) ? 1 : 0; // p5
                        neighbors[4] = (salida.getPixel(x, y + 1) == 0) ? 1 : 0;     // p6
                        neighbors[5] = (salida.getPixel(x - 1, y + 1) == 0) ? 1 : 0; // p7
                        neighbors[6] = (salida.getPixel(x - 1, y) == 0) ? 1 : 0;     // p8
                        neighbors[7] = (salida.getPixel(x - 1, y - 1) == 0) ? 1 : 0; // p9

                        int count = 0;
                        for (int n : neighbors)
                            count += n;
                        if (count < 2 || count > 6)
                            continue;

                        int transitions = 0;
                        for (int i = 0; i < 7; i++)
                        {
                            if (neighbors[i] == 0 && neighbors[i + 1] == 1)
                                transitions++;
                        }
                        if (neighbors[7] == 0 && neighbors[0] == 1)
                            transitions++;
                        if (transitions != 1)
                            continue;

                        if (neighbors[0] * neighbors[2] * neighbors[6] != 0)
                            continue;
                        if (neighbors[0] * neighbors[4] * neighbors[6] != 0)
                            continue;

                        marker[x][y] = true;
#pragma omp atomic // #pragma omp critical (Si funciona quedar atomic que es mas rapido)
                        change = true;
                    }
                }
#pragma omp for collapse(2) schedule(guided)
                for (int x = 1; x < width - 1; x++)
                    for (int y = 1; y < height - 1; y++)
                        if (marker[x][y])
                            salida.setPixel(x, y, 255);
            }
        } while (change);

        return salida;
    }
};

#endif