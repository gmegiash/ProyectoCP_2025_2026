#ifndef BUFFERIMAGE_H
#define BUFFERIMAGE_H

#include <vector>

class BufferImage {
private:
    int width;
    int height;
    // Usamos vector de vectores para simular char[][]
    // Usamos int para manejar los valores 0-255 cómodamente sin problemas de cast
    // Matriz dinamica que guarda los pixeles
    std::vector<std::vector<int>> img; 
    int max_val, min_val;
    float promedio;

public:
    BufferImage(int width, int height) : width(width), height(height) {
        // Inicializar la matriz [width][height]
        img.resize(width, std::vector<int>(height, 0));
    }

    int getHeight() const {
        return height;
    }

    int getWidth() const {
        return width;
    }

    void setPixel(int x, int y, int color) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            img[x][y] = color;
        }
    }

    int getPixel(int x, int y) const {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            return img[x][y];
        }
        return 0; // Retorno seguro
    }

    void setEstadisticas(int max_val, int min_val, float promedio) {
        this->max_val = max_val;
        this->min_val = min_val;
        this->promedio = promedio;
    }

    float getMedia() const {
        return promedio;
    }
};

#endif