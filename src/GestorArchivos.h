#ifndef GESTORARCHIVOS_H
#define GESTORARCHIVOS_H

#include <iostream>
#include <fstream> //Necesario para escribir archivos
#include <sstream> //Necesario para trocear los string
#include <string>
#include <vector>
#include "Minutia.h" //Necesitamos saber que es una Minutia para escribirla

class GestorArchivos
{
public:
    // 1.- Escribir en el archivo
    static bool guardarMinucias(const std::string &rutaArchivo, const std::vector<Minutia> &minucias)
    {
        // ios::app -> abre el archivo en modo "Aniadir", no borra lo que ya hay
        std::ofstream archivo(rutaArchivo, std::ios::app);

        if (!archivo.is_open())
        {
            std::cerr << "ERROR: No se pude crear o abrir el archivo TXT." << std::endl;
            return false;
        }

        for (size_t i = 0; i < minucias.size(); i++)
        {
            // Escribir formato: x, y, tipo, angulo...
            archivo << minucias[i].x << ", " << minucias[i].y << ", " << minucias[i].tipo;

            // Escribir los diferentes angulos separados por comas
            for (size_t j = 0; j < minucias[i].angulo.size(); j++)
            {
                archivo << ", " << minucias[i].angulo[j];
            }

            // Escribir el separador de minucias ";", excepto en la ultima minucia
            if (i < minucias.size() - 1)
            {
                archivo << "; ";
            }
        }
        // Al terminar la imagen damos un salto de linea
        archivo << "\n";

        archivo.close();
        return true;
    }

    // 2.- Leer del archivo a la memoria RAM
    static std::vector<std::vector<Minutia>> cargarBaseDatos(const std::string &rutaArchivo)
    {
        std::vector<std::vector<Minutia>> baseDatosRAM;
        std::ifstream archivo(rutaArchivo);

        if (!archivo.is_open())
        {
            std::cerr << "AVISO: No se encontro la base de datos previa. Se creara una nueva al guardar. " << std::endl;
            return baseDatosRAM;
        }

        std::string linea;
        // Leer linea por linea (cada linea es una imagen)
        while (std::getline(archivo, linea))
        {
            if (linea.empty())
                continue; // Si la linea esta vacia, saltarla

            std::vector<Minutia> minuciasImagen;
            std::stringstream ssLinea(linea);
            std::string tokenMinucia;

            // Cortamos la linea por ";" para obtener cada minucia individual
            while (std::getline(ssLinea, tokenMinucia, ';'))
            {
                if (tokenMinucia.empty())
                    continue;

                std::stringstream ssParam(tokenMinucia);
                std::string param;
                std::vector<std::string> parametros;

                // Cortamos la minucia por "," para obtener sus parametros
                while (std::getline(ssParam, param, ','))
                {
                    parametros.push_back(param);
                }

                // Reconstruimos el objeto Minucia a partir de los parametros
                if (parametros.size() >= 3)
                { //(x, y, tipo) como minimo
                    try
                    {
                        int x = std::stoi(parametros[0]);
                        int y = std::stoi(parametros[1]);
                        bool tipo = (std::stoi(parametros[2]) != 0); // Convertir a bool (0=false, cualquier otro valor=true)

                        std::vector<double> angulos;

                        // Lo que hay a partir de la posicion 3 son los angulos
                        for (size_t i = 3; i < parametros.size(); i++)
                        {
                            angulos.push_back(std::stod(parametros[i]));
                        }

                        // Guardamos la minucia en la lista de esta imagen
                        minuciasImagen.emplace_back(x, y, tipo, angulos);
                    }
                    catch (...)
                    {
                        // Si hay algun error al convertir un numero, ignora esa minucia y sigue
                    }
                }
            }
            // Guardamos la lista de minucias como una nueva "fila" en la matriz principal
            baseDatosRAM.push_back(minuciasImagen);
        }
        archivo.close();
        return baseDatosRAM;
    }
};

#endif