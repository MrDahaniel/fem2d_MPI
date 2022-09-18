#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mpi.h>

using namespace std;

int ProcNum;
int ProcRank;
int* pParallelPivotPos;
int* pProcPivotIter;
int* pProcInd;
int* pProcNum;

void RandomDataInitialization(double* pMatrix, double* pVector, int size) {
    int i, j;
    srand(time(0));
    for (i = 0; i < size; i++) {
        pVector[i] = rand() % 2000 / double(1000);
        for (j = 0; j < size; j++) {
            pMatrix[i * size + j] = rand() % 2000 / double(1000);
        }
    }
}

void ProcessInitialization(double*& pMatrix, double*& pVector,
    double*& pResult, double*& pProcRows, double*& pProcVector,
    double*& pProcResult, int& size, int& RowNum) {
    int RestRows;
    int i;

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    RestRows = size;


    for (i = 0; i < ProcRank; i++)
        RestRows = RestRows - RestRows / (ProcNum - i);

    RowNum = RestRows / (ProcNum - ProcRank);
    pProcRows = new double[RowNum * size];
    pProcVector = new double[RowNum];
    pProcResult = new double[RowNum];
    pParallelPivotPos = new int[size];
    pProcPivotIter = new int[RowNum];
    pProcInd = new int[ProcNum];
    pProcNum = new int[ProcNum];

    for (int i = 0; i < RowNum; i++)
        pProcPivotIter[i] = -1;
}

void DataDistribution(double* pMatrix, double* pProcRows, double* pVector,
    double* pProcVector, int size, int RowNum) {
    int* pSendNum;
    int* pSendInd;

    int RestRows = size;

    int i;

    pSendInd = new int[ProcNum];
    pSendNum = new int[ProcNum];

    RowNum = (size / ProcNum);
    pSendNum[0] = RowNum * size;
    pSendInd[0] = 0;
    for (i = 1; i < ProcNum; i++) {
        RestRows -= RowNum;
        RowNum = RestRows / (ProcNum - i);
        pSendNum[i] = RowNum * size;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
    }

    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE,
        pProcRows,
        pSendNum[ProcRank], MPI_DOUBLE, 0,
        MPI_COMM_WORLD);

    RestRows = size;
    pProcInd[0] = 0;
    pProcNum[0] = size / ProcNum;
    for (i = 1; i < ProcNum; i++) {
        RestRows -= pProcNum[i - 1];
        pProcNum[i] = RestRows / (ProcNum - i);
        pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
    }
    MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE,
        pProcVector,
        pProcNum[ProcRank], MPI_DOUBLE, 0,
        MPI_COMM_WORLD);

    delete[] pSendNum;
    delete[] pSendInd;
}

void ResultCollection(double* pProcResult, double* pResult) {

    MPI_Gatherv(pProcResult, pProcNum[ProcRank], MPI_DOUBLE, pResult,
        pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void PrintMatrix(double* pMatrix, int RowCount, int ColCount) {
    int i, j;
    for (i = 0; i < RowCount; i++) {
        for (j = 0; j < ColCount; j++)
            printf("%7.4f ", pMatrix[i * ColCount + j]);
        printf("\n");
    }
}

void PrintVector(double* pVector, int size) {
    int i;
    for (i = 0; i < size; i++)
        printf("%7.4f ", pVector[i]);
    printf("\n");
}

void PrintResultVector(double* pResult, int size) {
    int i;
    for (i = 0; i < size; i++)
        printf("%7.4f ", pResult[pParallelPivotPos[i]]);
    printf("\n");
}

void ParallelEliminateColumns(double* pProcRows, double* pProcVector,
    double* pPivotRow, int size, int RowNum, int Iter) {
    double multiplier;
    for (int i = 0; i < RowNum; i++) {
        if (pProcPivotIter[i] == -1) {
            multiplier = pProcRows[i * size + Iter] / pPivotRow[Iter];
            for (int j = Iter; j < size; j++) {
                pProcRows[i * size + j] -= pPivotRow[j] * multiplier;
            }
            pProcVector[i] -= pPivotRow[size] * multiplier;
        }
    }
}

void ParallelGaussianElimination(double* pProcRows, double* pProcVector,
    int size, int RowNum) {
    double MaxValue;
    int PivotPos;
    struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;


    double* pPivotRow = new double[size + 1];

    for (int i = 0; i < size; i++) {

        double MaxValue = 0;
        for (int j = 0; j < RowNum; j++) {
            if ((pProcPivotIter[j] == -1) && (MaxValue <
                fabs(pProcRows[j * size + i]))) {
                MaxValue = fabs(pProcRows[j * size + i]);
                PivotPos = j;
            }
        }
        ProcPivot.MaxValue = MaxValue;
        ProcPivot.ProcRank = ProcRank;

        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT,
            MPI_MAXLOC,
            MPI_COMM_WORLD);

        if (ProcRank == Pivot.ProcRank) {
            pProcPivotIter[PivotPos] = i;
            pParallelPivotPos[i] = pProcInd[ProcRank] + PivotPos;
        }
        MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank,
            MPI_COMM_WORLD);
        if (ProcRank == Pivot.ProcRank) {

            for (int j = 0; j < size; j++) {
                pPivotRow[j] = pProcRows[PivotPos * size + j];
            }
            pPivotRow[size] = pProcVector[PivotPos];
        }
        MPI_Bcast(pPivotRow, size + 1, MPI_DOUBLE,
            Pivot.ProcRank, MPI_COMM_WORLD);
        ParallelEliminateColumns(pProcRows, pProcVector,
            pPivotRow, size, RowNum, i);
    }
}

void FindBackPivotRow(int RowIndex, int size, int& IterProcRank, int
    & IterPivotPos) {
    for (int i = 0; i < ProcNum - 1; i++) {
        if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i + 1]))
            IterProcRank = i;
    }
    if (RowIndex >= pProcInd[ProcNum - 1])
        IterProcRank = ProcNum - 1;
    IterPivotPos = RowIndex - pProcInd[IterProcRank];
}

void ParallelBackSubstitution(double* pProcRows, double* pProcVector,
    double* pProcResult, int size, int RowNum) {
    int IterProcRank;
    int IterPivotPos;
    double IterResult;
    double val;

    for (int i = size - 1; i >= 0; i--) {
        FindBackPivotRow(pParallelPivotPos[i], size, IterProcRank,
            IterPivotPos);

        if (ProcRank == IterProcRank) {
            IterResult = pProcVector[IterPivotPos] /
                pProcRows[IterPivotPos * size + i];
            pProcResult[IterPivotPos] = IterResult;
        }

        MPI_Bcast(&IterResult, 1, MPI_DOUBLE, IterProcRank,
            MPI_COMM_WORLD);

        for (int j = 0; j < RowNum; j++)
            if (pProcPivotIter[j] < i) {
                val = pProcRows[j * size + i] * IterResult;
                pProcVector[j] = pProcVector[j] - val;
            }
    }
}

void ParallelResultCalculation(double* pProcRows, double* pProcVector,
    double* pProcResult, int size, int RowNum) {
    ParallelGaussianElimination(pProcRows, pProcVector, size,
        RowNum);
    ParallelBackSubstitution(pProcRows, pProcVector, pProcResult, size,
        RowNum);
}

void ProcessTermination(double* pProcRows, double* pProcVector, double* pProcResult) {
    delete[] pProcRows;
    delete[] pProcVector;
    delete[] pProcResult;
    delete[] pParallelPivotPos;
    delete[] pProcPivotIter;
    delete[] pProcInd;
    delete[] pProcNum;
}

void TestResult(double* pMatrix, double* pVector, double* pResult, int size) {
    double* pRightPartVector;

    int equal = 0;
    double Accuracy = 1.e-6;
    if (ProcRank == 0) {
        pRightPartVector = new double[size];
        for (int i = 0; i < size; i++) {
            pRightPartVector[i] = 0;
            for (int j = 0; j < size; j++) {
                pRightPartVector[i] += pMatrix[i * size + j] *
                    pResult[pParallelPivotPos[j]];
            }
        }
        for (int i = 0; i < size; i++) {
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

void gauss_solver(int size, double*& pMatrix, double*& pVector, double*& pResult) {
    double* pProcRows;
    double* pProcVector;
    double* pProcResult;
    int RowNum;
    double start, finish, duration;
    char* endptr;
    double t1, t2;

    setvbuf(stdout, 0, _IONBF, 0);

    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);

    if (ProcRank == 0) {
        t1 = MPI_Wtime();
    }

    // size = size * size;

    if (size < 1) {
        if (ProcRank == 0) {
            printf("Invalid args\n");
            printf("%d", MPI_COMM_WORLD);
        }
        MPI_Finalize();
    }

    if (ProcRank == 0)
        printf("Parallel Gauss algorithm for solving linear systems\n");

    ProcessInitialization(pMatrix, pVector, pResult,
        pProcRows, pProcVector, pProcResult, size, RowNum);

    DataDistribution(pMatrix, pProcRows, pVector, pProcVector, size,
        RowNum);

    ParallelResultCalculation(pProcRows, pProcVector, pProcResult,
        size, RowNum);

    ResultCollection(pProcResult, pResult);

    TestResult(pMatrix, pVector, pResult, size);

    if (ProcRank == 0) {
        t2 = MPI_Wtime();
    }

    // TestResult(pMatrix, pVector, pResult, size);
    if (ProcRank == 0) {
        printf("\nElapsed time for matrix %d is %f\n", size, t2 - t1);
    }

    ProcessTermination(pProcRows, pProcVector, pProcResult);
}



void exact(double x, double y, double* u, double* dudx, double* dudy)

{
    double pi = 3.141592653589793;

    *u = sin(pi * x) * sin(pi * y) + x;
    *dudx = pi * cos(pi * x) * sin(pi * y) + 1.0;
    *dudy = pi * sin(pi * x) * cos(pi * y);

    return;
}

double* r8ge_fs_new(int n, double a[], double b[])

{
    int i;
    int ipiv;
    int j;
    int jcol;
    double piv;
    double t;
    double* x;

    x = new double[n];

    for (i = 0; i < n; i++) {
        x[i] = b[i];
    }

    for (jcol = 1; jcol <= n; jcol++) {
        piv = fabs(a[jcol - 1 + (jcol - 1) * n]);
        ipiv = jcol;
        for (i = jcol + 1; i <= n; i++) {
            if (piv < fabs(a[i - 1 + (jcol - 1) * n])) {
                piv = fabs(a[i - 1 + (jcol - 1) * n]);
                ipiv = i;
            }
        }

        if (piv == 0.0) {
            cout << "\n";
            cout << "R8GE_FS_NEW - Fatal error!\n";
            cout << "  Zero pivot on step " << jcol << "\n";
            return NULL;
        }
        if (jcol != ipiv) {
            for (j = 1; j <= n; j++) {
                t = a[jcol - 1 + (j - 1) * n];
                a[jcol - 1 + (j - 1) * n] = a[ipiv - 1 + (j - 1) * n];
                a[ipiv - 1 + (j - 1) * n] = t;
            }
            t = x[jcol - 1];
            x[jcol - 1] = x[ipiv - 1];
            x[ipiv - 1] = t;
        }
        t = a[jcol - 1 + (jcol - 1) * n];
        a[jcol - 1 + (jcol - 1) * n] = 1.0;
        for (j = jcol + 1; j <= n; j++) {
            a[jcol - 1 + (j - 1) * n] = a[jcol - 1 + (j - 1) * n] / t;
        }
        x[jcol - 1] = x[jcol - 1] / t;
        for (i = jcol + 1; i <= n; i++) {
            if (a[i - 1 + (jcol - 1) * n] != 0.0) {
                t = -a[i - 1 + (jcol - 1) * n];
                a[i - 1 + (jcol - 1) * n] = 0.0;
                for (j = jcol + 1; j <= n; j++) {
                    a[i - 1 + (j - 1) * n] = a[i - 1 + (j - 1) * n] + t * a[jcol - 1 + (j - 1) * n];
                }
                x[i - 1] = x[i - 1] + t * x[jcol - 1];
            }
        }
    }
    for (jcol = n; 2 <= jcol; jcol--) {
        for (i = 1; i < jcol; i++) {
            x[i - 1] = x[i - 1] - a[i - 1 + (jcol - 1) * n] * x[jcol - 1];
        }
    }

    return x;
}

void timestamp()

{
#define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm* tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    cout << time_buffer << "\n";

    return;
#undef TIME_SIZE
}


int main(int argc, char* argsv[]) {
    int nx = 100;
    int ny = 100;

    double* a;
    double* b;
    double* c;

    int* element_node;

    double area, dqidx, dqidy, dqjdx, dqjdy, dudx, dudy;

    int e, element_num, i, i1, i2, i3, j, j2, k, node_num, nq1, nq2;
    int nti1, nti2, nti3, ntj1, ntj2, ntj3;
    int q1, q2, ti1, ti2, ti3, tj1, tj2, tj3;

    int rank;
    int size;

    const double pi = 3.141592653589793;

    double qi, rhs, u, wq, xq, yq;


    const double xl = 0.0;
    const double xr = 1.0;
    const double yb = 0.0;
    const double yt = 1.0;

    double* x;
    double* y;

    node_num = nx * ny;
    element_num = 2 * (nx - 1) * (ny - 1);
    element_node = new int[3 * element_num];

    timestamp();

    MPI_Init(&argc, &argsv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    if (rank == 0) {
        x = new double[node_num];
        y = new double[node_num];

        for (int idx = 0; idx < node_num; idx++) {
            j = 1 + idx / ny;
            i = 1 + idx % ny;

            x[idx] = ((double)(nx - i) * xl + (double)(i - 1) * xr) / (double)(nx - 1);
            y[idx] = ((double)(ny - j) * yb + (double)(j - 1) * yt) / (double)(ny - 1);
        }

        cout << "  Number of nodes =          " << node_num << "\n";
        cout << "  Number of elements =       " << element_num << "\n";

        k = 0;

        for (j = 1; j <= ny - 1; j++) {
            for (i = 1; i <= nx - 1; i++) {
                element_node[0 + k * 3] = i + (j - 1) * nx - 1;
                element_node[1 + k * 3] = i + 1 + (j - 1) * nx - 1;
                element_node[2 + k * 3] = i + j * nx - 1;
                k = k + 1;

                element_node[0 + k * 3] = i + 1 + j * nx - 1;
                element_node[1 + k * 3] = i + j * nx - 1;
                element_node[2 + k * 3] = i + 1 + (j - 1) * nx - 1;
                k = k + 1;
            }
        }

        a = new double[node_num * node_num];
        b = new double[node_num];
        c = new double[node_num];

        for (i = 0; i < node_num; i++) {
            b[i] = 0.0;
        }
        for (j = 0; j < node_num; j++) {
            for (i = 0; i < node_num; i++) {
                a[i + j * node_num] = 0.0;
            }
        }

        for (e = 0; e < element_num; e++) {
            i1 = element_node[0 + e * 3];
            i2 = element_node[1 + e * 3];
            i3 = element_node[2 + e * 3];
            area = 0.5 *
                (x[i1] * (y[i2] - y[i3]) + x[i2] * (y[i3] - y[i1]) + x[i3] * (y[i1] - y[i2]));
            for (q1 = 0; q1 < 3; q1++) {
                q2 = (q1 + 1) % 3;

                nq1 = element_node[q1 + e * 3];
                nq2 = element_node[q2 + e * 3];

                xq = 0.5 * (x[nq1] + x[nq2]);
                yq = 0.5 * (y[nq1] + y[nq2]);
                wq = 1.0 / 3.0;
                for (ti1 = 0; ti1 < 3; ti1++) {
                    ti2 = (ti1 + 1) % 3;
                    ti3 = (ti1 + 2) % 3;

                    nti1 = element_node[ti1 + e * 3];
                    nti2 = element_node[ti2 + e * 3];
                    nti3 = element_node[ti3 + e * 3];

                    qi = 0.5 * ((x[nti3] - x[nti2]) * (yq - y[nti2]) - (y[nti3] - y[nti2]) * (xq - x[nti2])) / area;
                    dqidx = -0.5 * (y[nti3] - y[nti2]) / area;
                    dqidy = 0.5 * (x[nti3] - x[nti2]) / area;

                    rhs = 2.0 * pi * pi * sin(pi * xq) * sin(pi * yq);

                    b[nti1] = b[nti1] + area * wq * rhs * qi;
                    for (tj1 = 0; tj1 < 3; tj1++) {
                        tj2 = (tj1 + 1) % 3;
                        tj3 = (tj1 + 2) % 3;

                        ntj1 = element_node[tj1 + e * 3];
                        ntj2 = element_node[tj2 + e * 3];
                        ntj3 = element_node[tj3 + e * 3];

                        dqjdx = -0.5 * (y[ntj3] - y[ntj2]) / area;
                        dqjdy = 0.5 * (x[ntj3] - x[ntj2]) / area;

                        a[nti1 + ntj1 * node_num] = a[nti1 + ntj1 * node_num] + area * wq * (dqidx * dqjdx + dqidy * dqjdy);
                    }
                }
            }
        }
        k = 0;
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                if (i == 1 || i == nx || j == 1 || j == ny) {
                    exact(x[k], y[k], &u, &dudx, &dudy);
                    for (j2 = 0; j2 < node_num; j2++) {
                        a[k + j2 * node_num] = 0.0;
                    }
                    a[k + k * node_num] = 1.0;
                    b[k] = u;
                }
                k = k + 1;
            }
        }
    }


    MPI_Barrier(MPI_COMM_WORLD);

    // c = r8ge_fs_new(node_num, a, b);
    gauss_solver(node_num, a, b, c);

    if (rank == 0) {
        cout << "\n";
        cout << "     K     I     J          X           Y        U               U                Error\n";
        cout << "                                                 exact           computed\n";
        cout << "\n";

        k = 0;

        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                exact(x[k], y[k], &u, &dudx, &dudy);

                cout << "  " << setw(4) << k
                    << "  " << setw(4) << i
                    << "  " << setw(4) << j
                    << "  " << setw(10) << x[k]
                    << "  " << setw(10) << y[k]
                    << "  " << setw(14) << u
                    << "  " << setw(14) << c[k]
                    << "  " << setw(14) << fabs(u - c[k]) << "\n";

                k = k + 1;
            }
            cout << "\n";
        }
        delete[] a;
        delete[] b;
        delete[] c;
        delete[] element_node;
        delete[] x;
        delete[] y;
        cout << "\n";
        cout << "FEM2D_POISSON_RECTANGLE_LINEAR:\n";
        cout << "  Normal end of execution.\n";
        cout << "\n";
        timestamp();
    }

    MPI_Finalize();

    return 0;
}