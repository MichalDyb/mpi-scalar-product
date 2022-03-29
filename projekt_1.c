#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

//   Dla środowiska gdzie nie została ustalona relacja pomiędzy procesami
//  kompilacja   mpicc -o projekt_1 projekt_1.c
//  uruchamianie mpirun --host localhost:liczba_procesorów/wątków ./projekt_1

int main(int argc, char ** argv)
{
    //Deklaracja i incjalizacja zmienych
    double *vt_1, *vt_1_part, *vt_2, *vt_2_part, *scores, score = 0; // Wskaźniki do tablic liczb typu double (nasze wektory i pod_wektory) i tablicy na wyniki oraz sumy dla każdego procesu z osobna
    int vt_normal_size; // Zmienna przechowująca ilość elementów dla każdego wektora, jest ona inicjalizowana przez użytkownika
    int vt_extend_size; // Zmienna przechowująca ilość elementów (równych 0), o które został rozszerzony wektor
    int vt_real_size; // Zmienna przechowująca realną ilość elementów dla każdego wektora
    int vt_part_size; // Zmienna przechowująca ilość elementów wektorów jaki przypada na każdy z procesów
    int rank; // Numer mojego procesu
    int size; // Liczba procesów
    struct timespec start, end; // Zmienna przechowująca czas rozpoczęcia i końca obliczeń

    MPI_Init(&argc, &argv); // Rozpoczęcie obliczeń MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Pobranie aktualnego numeru procesu
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Pobranie liczby procesów

    if(rank == 0) // Instrukcje wykonywane dla głównego procesu programu (proces root)
    {
        fprintf(stderr, "Podaj długośc wektorów (liczba całkowita większa od 0): "); // Wyświetlenie komunikatu z prośbą o podanie długości wektora
        scanf("%d", &vt_normal_size); // Pobranie długości wektów od użytkownika

        if(vt_normal_size < 0) // Wyjście z programu dla wektora mniejszego niż 1
        {
            printf("Podany rozmiar wektorów jest zbyt mały, ilość elementów jest mniejsza niż ilość procesów: %d.\n", size);
            MPI_Finalize(); // Koniec obliczeń MPI 
            exit(1); // Wyjście z programu z kodem błędu
        }

        vt_extend_size = size - (vt_normal_size % size); // Obliczenie o ile elementów należy rozszerzyć wektor, aby można było podzielić je równo dla każdego procesu
        vt_real_size = vt_normal_size + vt_extend_size; // Obliczenie realnej długości wektorów
        vt_part_size = vt_real_size / size; // Obliczenie ilości elementów przypadających na każdy proces
        vt_1 = malloc(vt_real_size * sizeof(double)); // Zaalokowanie pamięci na pierwszy wektor
        vt_2 = malloc(vt_real_size * sizeof(double)); // Zaalokowanie pamięci na drugi wektor
        vt_1_part = malloc(vt_part_size * sizeof(double)); // Zaalokowanie pamięci na pierwszy wektor częsciowy
        vt_2_part = malloc(vt_part_size * sizeof(double)); // Zaalokowanie pamięci na drugi wektor częściowy
        scores = malloc(size * sizeof(double)); // Zaalokowanie pamięci na tablice wyników

        for(int i = 0; i < vt_real_size; i++) // Inicjalizacja wektorów liczbami losowymi
        {
            if(i < vt_normal_size) // Wypełnienie elementów wektorów liczbami typu rzeczywistego z przedziału (-25, 25) 
            {
                srand(time(NULL) + rand());
                vt_1[i] = (double) rand() / RAND_MAX * 50 - 25;
                vt_2[i] = (double) rand() / RAND_MAX * 50 - 25;
            }
            else // Dla dodatkowo wygenerowanych elementów wartość wynosi 0, aby wynik obliczeń nie został zakłamany
            {
                vt_1[i] = 0;
                vt_2[i] = 0;
            }
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &start); // Pobranie czasu rozpoczęcia obliczeń
        MPI_Bcast(&vt_real_size, 1, MPI_INT, 0, MPI_COMM_WORLD); // Wysłanie rozmiaru wektora
        MPI_Bcast(&vt_part_size, 1, MPI_INT, 0, MPI_COMM_WORLD); // Wysłanie rozmiaru części wektora
        MPI_Scatter(vt_1, vt_part_size, MPI_DOUBLE, vt_1_part, vt_part_size, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Wysłanie podzielonego wektora 1 na kawałki do procesów w grupie
        MPI_Scatter(vt_2, vt_part_size, MPI_DOUBLE, vt_2_part, vt_part_size, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Wysłanie podzielonego wektora 1 na kawałki do procesów w grupie
        MPI_Barrier(MPI_COMM_WORLD); // Bariera synchronizacyjna

        for (int i = 0; i < vt_part_size; i++) // Obliczenie iloczynu częsciowych wektorów
        {
            score += (vt_1[i] * vt_2[i]);
        }
        fprintf(stderr, "Iloczyn skalarny procesora %d wynosi: %f\n", rank, score); // Wyświetlenie informacji o obliczonym częściowym iloczynie skalarnym
        MPI_Barrier(MPI_COMM_WORLD); // Bariera synchronizacyjna
        MPI_Gather(&score, 1, MPI_DOUBLE, scores, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Zebranie wyników z wszystkich procesów grupie

        score = 0; // Wyzerowanie iloczynu skalarnego
        for (int i = 0; i < size; i++) // Połączenie wyników częściowych
        {
            score += scores[i];
        }

        fprintf(stderr, "Iloczyn skalarny wektorów %d-elementowych dla %d procesorów wynosi: %f\n", vt_normal_size, size, score); // Wyświetlenie informacji o obliczonym iloczynie skalarnym

        free(vt_1); // Zwolnienie obszaru pamięci wektora 1 
        free(vt_2); // Zwolnienie obszaru pamięci wektora 2
        free(vt_1_part); // Zwolnienie obszaru pamięci wektora częściowego 1 
        free(vt_2_part); // Zwolnienie obszaru pamięci wektora częściowego 2
        free(scores); // Zwolnienie obszaru pamięci tablicy wyników
    }
    else // Instrukcje wykonywane przez procesy podległe
    {
        MPI_Barrier(MPI_COMM_WORLD); // Bariera synchronizacyjna
        MPI_Bcast(&vt_real_size, 1, MPI_INT, 0, MPI_COMM_WORLD); // Odebranie rozmiaru wektora
        MPI_Bcast(&vt_part_size, 1, MPI_INT, 0, MPI_COMM_WORLD); // Odebranie rozmiaru części wektora
        vt_1 = malloc(vt_real_size * sizeof(double)); // Zaalokowanie pamięci na pierwszy wektor
        vt_2 = malloc(vt_real_size * sizeof(double)); // Zaalokowanie pamięci na drugi wektor
        vt_1_part = malloc(vt_part_size * sizeof(double)); // Zaalokowanie pamięci na pierwszy wektor częsciowy
        vt_2_part = malloc(vt_part_size * sizeof(double)); // Zaalokowanie pamięci na drugi wektor częściowy

        MPI_Scatter(vt_1, vt_part_size, MPI_DOUBLE, vt_1_part, vt_part_size, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Odebranie podzielonego wektora 1 na kawałki
        MPI_Scatter(vt_2, vt_part_size, MPI_DOUBLE, vt_2_part, vt_part_size, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Odebranie podzielonego wektora 1 na kawałki
        for (int i = 0; i < vt_part_size; i++) // Obliczenie iloczynu częsciowych wektorów
        {
            score += (vt_1_part[i] * vt_2_part[i]);
        }
        fprintf(stderr, "Iloczyn skalarny procesora %d wynosi: %f\n", rank, score); // Wyświetlenie informacji o obliczonym częściowym iloczynie skalarnym
        MPI_Barrier(MPI_COMM_WORLD); // Bariera synchronizacyjna
        MPI_Gather(&score, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Wysłanie wyników do procesu zbierającego wyniki

        free(vt_1); // Zwolnienie obszaru pamięci wektora 1 
        free(vt_2); // Zwolnienie obszaru pamięci wektora 2
        free(vt_1_part); // Zwolnienie obszaru pamięci wektora częściowego 1 
        free(vt_2_part); // Zwolnienie obszaru pamięci wektora częściowego 2
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end); // Pobranie informacji o czasie zakończenia obliczeń
    MPI_Finalize(); // Koniec obliczeń MPI 

    if(rank == 0)
    {
        u_int64_t delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000; // Obliczenie różnicy między czasem końca obliczeń, a rozpoczęciem w mikrosekundach
        fprintf(stderr, "Obliczenia trwały %6.2f milisekund\n", (double) delta_us / 1000); // Wyświetlenie informacji na temat czasu obliczeń
    }
    
    return 0; // Wyjście z programu
}