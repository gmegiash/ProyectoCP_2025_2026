#ifndef MINUTIA_H
#define MINUTIA_H

#include <vector>

struct Minutia {
    int x, y;
    bool tipo;
    std::vector<double> angulo;

    Minutia(int x, int y, bool tipo, std::vector<double> angulo)
        : x(x), y(y), tipo(tipo), angulo(angulo) {}
};

#endif