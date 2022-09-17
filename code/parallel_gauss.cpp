#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
int ProcNum; // Количество доступных процессов
int ProcRank; // Ранг текущего процесса
int* pParallelPivotPos; // Количество строк, выбранных в качестве опорных
int* pProcPivotIter; // Количество итераций, при которых строки процессора //использовались в качестве опорных
int* pProcInd; // Номер первой строки, расположенной в процессах
int* pProcNum; // Число строк СЛАУ, расположенных в процессах
// Функция случайного определения элементов матрицы и вектора
void RandomDataInitialization(double* pMatrix, double* pVector, int Size) {
    int i, j; // Loop variables
    srand(time(0));
    for (i = 0; i < Size; i++) {
        pVector[i] = rand() % 2000 / double(1000);
        for (j = 0; j < Size; j++) {
            pMatrix[i * Size + j] = rand() % 2000 / double(1000);
        }
    }
}
// Функция для выделения памяти и инициализации данных
void ProcessInitialization(double*& pMatrix, double*& pVector,
    double*& pResult, double*& pProcRows, double*& pProcVector,
    double*& pProcResult, int& Size, int& RowNum) {
    int RestRows; // Количество строк, которые еще не были распределены
    int i; // Циклическая переменная
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    RestRows = Size;
    for (i = 0; i < ProcRank; i++)
        RestRows = RestRows - RestRows / (ProcNum - i);
    RowNum = RestRows / (ProcNum - ProcRank);
    pProcRows = new double[RowNum * Size];
    pProcVector = new double[RowNum];
    pProcResult = new double[RowNum];
    pParallelPivotPos = new int[Size];
    pProcPivotIter = new int[RowNum];
    pProcInd = new int[ProcNum];
    pProcNum = new int[ProcNum];
    for (int i = 0; i < RowNum; i++)
        pProcPivotIter[i] = -1;
    if (ProcRank == 0) {
        pMatrix = new double[Size * Size];
        pVector = new double[Size];
        pResult = new double[Size];
        RandomDataInitialization(pMatrix, pVector, Size);
    }
}
// Функция распределения данных между процессами
void DataDistribution(double* pMatrix, double* pProcRows, double* pVector,
    double* pProcVector, int Size, int RowNum) {
    int* pSendNum; // Количество элементов, отправленных в обработку
    int* pSendInd; // Индекс первого элемента данных отправленного
    // к процессу
    int RestRows = Size; // Количество строк, которые не были
    // распределены
    int i; // Циклическая переменная
    // Память выделенная для временных объектов
    pSendInd = new int[ProcNum];
    pSendNum = new int[ProcNum];
    // Определение расположения строк матрицы для текущего процесса
    RowNum = (Size / ProcNum);
    pSendNum[0] = RowNum * Size;
    pSendInd[0] = 0;
    for (i = 1; i < ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows / (ProcNum - i);
        pSendNum[i] = RowNum * Size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
    }
    // Прохождение по строкам
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE,
        pProcRows,
        pSendNum[ProcRank], MPI_DOUBLE, 0,
        MPI_COMM_WORLD);
    // Определение расположения строк матрицы для текущего процесса
    RestRows = Size;
    pProcInd[0] = 0;
    pProcNum[0] = Size / ProcNum;
    for (i = 1; i < ProcNum; i++) {
        RestRows -= pProcNum[i - 1];
        pProcNum[i] = RestRows / (ProcNum - i);
        pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
    }
    MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE,
        pProcVector,
        pProcNum[ProcRank], MPI_DOUBLE, 0,
        MPI_COMM_WORLD);
    // Освобождение памяти
    delete[] pSendNum;
    delete[] pSendInd;
}
// Функция для сбора вектора результата
void ResultCollection(double* pProcResult, double* pResult) {
    // Сбор всего вектора результата на каждом процессоре
    MPI_Gatherv(pProcResult, pProcNum[ProcRank], MPI_DOUBLE, pResult,
        pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
// Функция для форматированного вывода матрицы
void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
    int i, j; // Loop variables
    for (i = 0; i < RowCount; i++) {
        for (j = 0; j < ColCount; j++)
            printf("%7.4f ", pMatrix[i * ColCount + j]);
        printf("\n");
    }
}
// Функция для форматированного вывода вектора
void PrintVector(double* pVector, int Size) {
    int i;
    for (i = 0; i < Size; i++)
        printf("%7.4f ", pVector[i]);
    printf("\n");
}
// Функция форматирования выходного результата вектора
void PrintResultVector(double* pResult, int Size) {
    int i;
    for (i = 0; i < Size; i++)
        printf("%7.4f ", pResult[pParallelPivotPos[i]]);
    printf("\n");
}
// Функция исключения столбца
void ParallelEliminateColumns(double* pProcRows, double* pProcVector,
    double* pPivotRow, int Size, int RowNum, int Iter) {
    double multiplier;
    for (int i = 0; i < RowNum; i++) {
        if (pProcPivotIter[i] == -1) {
            multiplier = pProcRows[i * Size + Iter] / pPivotRow[Iter];
            for (int j = Iter; j < Size; j++) {
                pProcRows[i * Size + j] -= pPivotRow[j] * multiplier;
            }
            pProcVector[i] -= pPivotRow[Size] * multiplier;
        }
    }
}
// Функция исключения Гаусса
void ParallelGaussianElimination(double* pProcRows, double* pProcVector,
    int Size, int RowNum) {
    double MaxValue; // Значение опорного элемента этого процесса
    int PivotPos; // Расположение строки в строке процесса
    struct { double MaxValue; int ProcRank; } ProcPivot, Pivot; //Структура для выбора строки
    // pPivotRow используется для хранения строки сводки и соответствующей
    // элемент вектора b
    double* pPivotRow = new double[Size + 1];
    // Итерации на стадии исключерния функцией Гаусса
    for (int i = 0; i < Size; i++) {
        // Вычисление строки локального поворота
        double MaxValue = 0;
        for (int j = 0; j < RowNum; j++) {
            if ((pProcPivotIter[j] == -1) && (MaxValue <
                fabs(pProcRows[j * Size + i]))) {
                MaxValue = fabs(pProcRows[j * Size + i]);
                PivotPos = j;
            }
        }
        ProcPivot.MaxValue = MaxValue;
        ProcPivot.ProcRank = ProcRank;
        // Нахождение максимального значения для MaxValue
        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT,
            MPI_MAXLOC,
            MPI_COMM_WORLD);
        // Передача строки
        if (ProcRank == Pivot.ProcRank) {
            pProcPivotIter[PivotPos] = i; //iteration number
            pParallelPivotPos[i] = pProcInd[ProcRank] + PivotPos;
        }
        MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank,
            MPI_COMM_WORLD);
        if (ProcRank == Pivot.ProcRank) {
            // Fill the pivot row
            for (int j = 0; j < Size; j++) {
                pPivotRow[j] = pProcRows[PivotPos * Size + j];
            }
            pPivotRow[Size] = pProcVector[PivotPos];
        }
        MPI_Bcast(pPivotRow, Size + 1, MPI_DOUBLE,
            Pivot.ProcRank, MPI_COMM_WORLD);
        ParallelEliminateColumns(pProcRows, pProcVector,
            pPivotRow, Size, RowNum, i);
    }
}
// Функция получения строки для обратного хода метода Гаусса
void FindBackPivotRow(int RowIndex, int Size, int& IterProcRank, int
    & IterPivotPos) {
    for (int i = 0; i < ProcNum - 1; i++) {
        if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i + 1]))
            IterProcRank = i;
    }
    if (RowIndex >= pProcInd[ProcNum - 1])
        IterProcRank = ProcNum - 1;
    IterPivotPos = RowIndex - pProcInd[IterProcRank];
}
// Функция для обратного хода метода Гаусса
void ParallelBackSubstitution(double* pProcRows, double* pProcVector,
    double* pProcResult, int Size, int RowNum) {
    int IterProcRank; // Rank of the process with the current pivot row
    int IterPivotPos; // Position of the pivot row of the process
    double IterResult; // Calculated value of the current unknown
    double val;
    // Итерационный этап обратного хода
    for (int i = Size - 1; i >= 0; i--) {
        FindBackPivotRow(pParallelPivotPos[i], Size, IterProcRank,
            IterPivotPos);
        // Расчѐт неизвестных переменных
        if (ProcRank == IterProcRank) {
            IterResult = pProcVector[IterPivotPos] /
                pProcRows[IterPivotPos * Size + i];
            pProcResult[IterPivotPos] = IterResult;
        }
        // Передача текущей неизвестной
        MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterProcRank,
            MPI_COMM_WORLD);
        // Обновление значений вектора b
        for (int j = 0; j < RowNum; j++)
            if (pProcPivotIter[j] < i) {
                val = pProcRows[j * Size + i] * IterResult;
                pProcVector[j] = pProcVector[j] - val;
            }
    }
}
void TestDistribution(double* pMatrix, double* pVector, double* pProcRows,
    double* pResult, int Size, int RowNum) {
    if (ProcRank == 0) {
        printf("Initial Matrix: \n");
        PrintMatrix(pMatrix, Size, Size);
        printf("Initial Vector: \n");
        PrintVector(pVector, Size);
        printf("Result: \n");
        PrintResultVector(pResult, Size);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
// Вызов параллельного алгоритма Гаусса
void ParallelResultCalculation(double* pProcRows, double* pProcVector,
    double* pProcResult, int Size, int RowNum) {
    ParallelGaussianElimination(pProcRows, pProcVector, Size,
        RowNum);
    ParallelBackSubstitution(pProcRows, pProcVector, pProcResult, Size,
        RowNum);
}
// Функция для процесса очищения выделенной памяти
void ProcessTermination(double* pMatrix, double* pVector, double* pResult,
    double* pProcRows, double* pProcVector, double* pProcResult) {
    if (ProcRank == 0) {
        delete[] pMatrix;
        delete[] pVector;
        delete[] pResult;
    }
    delete[] pProcRows;
    delete[] pProcVector;
    delete[] pProcResult;
    delete[] pParallelPivotPos;
    delete[] pProcPivotIter;
    delete[] pProcInd;
    delete[] pProcNum;
}
// Функция для проверки результата
void TestResult(double* pMatrix, double* pVector, double* pResult, int Size) {
    /* Указатель для хранения вектора, который является
    результатом суммы
    произведения матрицы СЛАУ и вектора корней */
    double* pRightPartVector;
    // Флаг, показывающий что части векторов идентичны или нет
    int equal = 0;
    double Accuracy = 1.e-6; // Точность решения
    if (ProcRank == 0) {
        pRightPartVector = new double[Size];
        for (int i = 0; i < Size; i++) {
            pRightPartVector[i] = 0;
            for (int j = 0; j < Size; j++) {
                pRightPartVector[i] += pMatrix[i * Size + j] *
                    pResult[pParallelPivotPos[j]];
            }
        }
        for (int i = 0; i < Size; i++) {
            if (fabs(pRightPartVector[i] - pVector[i]) > Accuracy)
                equal = 1;
        }
        if (equal == 1)
            printf("The result of the parallel Gauss algorithm is NOT correct. Check your code.\n");
        else
            printf("The result of the parallel Gauss algorithm is correct.\n");
        delete[] pRightPartVector;
    }
}
int main(int argc, char* argv[]) {
    double* pMatrix; // Матрица линейных уравнений
    double* pVector; // Правая часть линейных уравнений
    double* pResult; // Вектор результата
    double* pProcRows; // Строки матрицы А
    double* pProcVector; // Вектор b
    double* pProcResult; // Результирующий вектор
    int Size; // Размерность матрицы
    int RowNum; // Количество строк матрицы
    double start, finish, duration;
    char* endptr;
    double t1, t2;
    setvbuf(stdout, 0, _IONBF, 0);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    if (ProcRank == 0) {
        t1 = MPI_Wtime();
    }
    //if (argc < 2) {
    //if (ProcRank == 0) {
    //printf("Invalid args\n");
    //printf("%d", MPI_COMM_WORLD);
    //}
    //MPI_Finalize();
    //return 0;
    //}
    Size = 120 * 120;

    if (Size < 1) {
        if (ProcRank == 0) {
            printf("Invalid args\n");
            printf("%d", MPI_COMM_WORLD);
        }
        MPI_Finalize();
        return 0;
    }
    if (ProcRank == 0)
        printf("Parallel Gauss algorithm for solving linear systems\n");
    // Выделение памяти и инициализация данных
    ProcessInitialization(pMatrix, pVector, pResult,
        pProcRows, pProcVector, pProcResult, Size, RowNum);
    // Выполнение алгоритма Гаусса
    DataDistribution(pMatrix, pProcRows, pVector, pProcVector, Size,
        RowNum);
    ParallelResultCalculation(pProcRows, pProcVector, pProcResult,
        Size, RowNum);
    ResultCollection(pProcResult, pResult);
    if (ProcRank == 0) {
        t2 = MPI_Wtime();
    }
    if (Size < 11) {
        TestDistribution(pMatrix, pVector, pProcRows, pResult, Size,
            RowNum);
    }
    TestResult(pMatrix, pVector, pResult, Size);
    if (ProcRank == 0) {
        printf("\nElapsed time for matrix %d is %f\n", Size, t2 - t1);
    }
    // Освобождение памяти
    ProcessTermination(pMatrix, pVector, pResult, pProcRows, pProcVector, pProcResult);
    MPI_Finalize();
    return 0;
}

// https://gist.github.com/cxspxr/cbf992046eb859283736e7132c8523d3