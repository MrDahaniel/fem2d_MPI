#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mpi.h>

#define ORCHESTRATOR 0

#define X_TAG 10
#define Y_TAG 20

#define LOWER_TAG 100
#define UPPER_TAG 200

using namespace std;

void exact(double x, double y, double* u, double* dudx, double* dudy);
double* r8ge_fs_new(int n, double a[], double b[]);
void timestamp();
int* evenly_divide(int to_be_divided, int parts, int lower_bound = 0);


int main(int argc, char* argsv[]) {
    int rank, size;
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argsv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const double pi = 3.141592653589793;

    int nx = 15;
    int ny = 10;

    double area;

    double dqidx, dqidy, dqjdx, dqjdy, dudx, dudy;

    int e, i, i1, i2, i3, j, j2, nq1, nq2, nti1, nti2, nti3, ntj1, ntj2, ntj3;

    int q1, q2;
    double qi;

    double xq;
    double yq;

    int ti1, ti2, ti3, tj1, tj2, tj3;

    double rhs, u, wq;

    const double xl = 0.0;
    const double xr = 1.0;
    const double yb = 0.0;
    const double yt = 1.0;

    const int node_num = nx * ny;

    // double* x = new double[node_num];
    // double* y = new double[node_num];

    int element_num = 2 * (nx - 1) * (ny - 1);

    double* a = new double[node_num * node_num];
    double* b = new double[node_num];

    if (rank == 0) {
        {
            timestamp();
            cout << "\n";
            cout << "FEM2D_POISSON_RECTANGLE_LINEAR\n";
            cout << "   C++ version, now with 100% more MPI!\n";
            cout << "\n";
            cout << "   Solution of the Poisson equation:\n";
            cout << "\n";
            cout << "   - Uxx - Uyy = F(x,y) inside the region,\n";
            cout << "       U(x,y) = G(x,y) on the boundary of the region.\n";
            cout << "\n";
            cout << "   The region is a rectangle, defined by:\n";
            cout << "\n";
            cout << "   " << xl << " = XL<= X <= XR = " << xr << "\n";
            cout << "   " << yb << " = YB<= Y <= YT = " << yt << "\n";
            cout << "\n";
            cout << "   The finite element method is used, with piecewise\n";
            cout << "   linear basis functions on 3 node triangular\n";
            cout << "   elements.\n";
            cout << "\n";
            cout << "   The corner nodes of the triangles are generated by an\n";
            cout << "   underlying grid whose dimensions are\n";
            cout << "\n";
            cout << "   NX =                       " << nx << "\n";
            cout << "   NY =                       " << ny << "\n";
            cout << "   Number of nodes =          " << node_num << "\n";
            cout << "   Number of elements =       " << element_num << "\n";
        }

        int* element_node = new int[3 * element_num];

        double* x = new double[node_num];
        double* y = new double[node_num];

        // The first operation is to divide our ny and nx evenly between our compute ranks
        int* node_splits = evenly_divide(node_num, size - 1);

        printf("splits: ");
        for (int i = 0; i < size; i++) {
            printf("%d ", node_splits[i]);
        }
        printf("\n");

        // Then we send the ranges to be calculated
        for (int rank_id = 1; rank_id < size; rank_id++) {
            // Sending the node_num range
            MPI_Send(&node_splits[rank_id - 1], 1, MPI_INT, rank_id, LOWER_TAG, MPI_COMM_WORLD);
            MPI_Send(&node_splits[rank_id], 1, MPI_INT, rank_id, UPPER_TAG, MPI_COMM_WORLD);
        }

        // And we get the ranges that were calculated.
        int k = 0;
        for (int rank_id = 1; rank_id < size; rank_id++) {
            int lower = node_splits[rank_id - 1] + 1;
            int upper = node_splits[rank_id];

            int inner_node_num = upper - lower + 1;

            double* partial_x = new double[inner_node_num];
            double* partial_y = new double[inner_node_num];

            MPI_Recv(partial_x, inner_node_num, MPI_DOUBLE, rank_id, X_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(partial_y, inner_node_num, MPI_DOUBLE, rank_id, Y_TAG, MPI_COMM_WORLD, &status);

            printf("partial_x: ");
            for (int i = 0; i < inner_node_num; i++) {
                printf("%.2f ", partial_x[i]);
            }

            for (int idx = 0; idx < inner_node_num; idx++) {
                x[k] = partial_x[idx];
                y[k] = partial_y[idx];
                k += 1;
            }

            delete[] partial_x;
            delete[] partial_y;
        }


        int k = 0;

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

        {
            //
            //  Assemble the coefficient matrix A and the right-hand side B of the
            //  finite element equations, ignoring boundary conditions.
            //

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
                //
                //  Consider each quadrature point.
                //  Here, we use the midside nodes as quadrature points.
                //
                for (q1 = 0; q1 < 3; q1++) {
                    q2 = (q1 + 1) % 3;

                    nq1 = element_node[q1 + e * 3];
                    nq2 = element_node[q2 + e * 3];

                    xq = 0.5 * (x[nq1] + x[nq2]);
                    yq = 0.5 * (y[nq1] + y[nq2]);
                    wq = 1.0 / 3.0;
                    //
                    //  Consider each test function in the element.
                    //
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
                        //
                        //  Consider each basis function in the element.
                        //
                        for (tj1 = 0; tj1 < 3; tj1++) {
                            tj2 = (tj1 + 1) % 3;
                            tj3 = (tj1 + 2) % 3;

                            ntj1 = element_node[tj1 + e * 3];
                            ntj2 = element_node[tj2 + e * 3];
                            ntj3 = element_node[tj3 + e * 3];

                            //        qj = 0.5 * (
                            //            ( x[ntj3] - x[ntj2] ) * ( yq - y[ntj2] )
                            //          - ( y[ntj3] - y[ntj2] ) * ( xq - x[ntj2] ) ) / area;
                            dqjdx = -0.5 * (y[ntj3] - y[ntj2]) / area;
                            dqjdy = 0.5 * (x[ntj3] - x[ntj2]) / area;

                            a[nti1 + ntj1 * node_num] = a[nti1 + ntj1 * node_num] + area * wq * (dqidx * dqjdx + dqidy * dqjdy);
                        }
                    }
                }
            }
            //
            //  BOUNDARY CONDITIONS
            //
            //  If the K-th variable is at a boundary node, replace the K-th finite
            //  element equation by a boundary condition that sets the variable to U(K).
            //
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
            //  SOLVE the linear system A * C = B.
            //
            //  The solution is returned in C.
            //
            double* c = r8ge_fs_new(node_num, a, b);
            //
            //  COMPARE computed and exact solutions.
            //
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
            //
            //  Deallocate memory.
            //
            delete[] a;
            delete[] b;
            delete[] c;
            delete[] element_node;
            delete[] x;
            delete[] y;
            //
            //  Terminate.
            //
            cout << "\n";
            cout << "FEM2D_POISSON_RECTANGLE_LINEAR:\n";
            cout << "  Normal end of execution.\n";
            cout << "\n";
            timestamp();
        }

    } else {

        // First scope: Calculating x and y
        {
            int node_lower, node_upper;

            // Recieve the lower and upper bounds
            MPI_Recv(&node_lower, 1, MPI_INT, ORCHESTRATOR, LOWER_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&node_upper, 1, MPI_INT, ORCHESTRATOR, UPPER_TAG, MPI_COMM_WORLD, &status);

            int inner_node_num = node_upper - node_lower;

            double* x = new double[inner_node_num];
            double* y = new double[inner_node_num];

            int k = 0;
            for (int idx = node_lower; idx < node_upper; idx++) {
                int j = 1 + (idx / ny);
                int i = 1 + (idx % ny);

                x[k] = ((double)(nx - i) * xl + (double)(i - 1) * xr) / (double)(nx - 1);
                y[k] = ((double)(ny - j) * yb + (double)(j - 1) * yt) / (double)(ny - 1);
                k = k + 1;
            }

            printf("lower: %d, upper: %d rank %d x: ", node_lower, node_upper, rank);
            for (int i = 0; i < inner_node_num; i++) {
                printf("%.2f ", x[i]);

            }
            printf("\n");

            // We return the values calculated 
            MPI_Send(x, inner_node_num, MPI_DOUBLE, ORCHESTRATOR, X_TAG, MPI_COMM_WORLD);
            MPI_Send(y, inner_node_num, MPI_DOUBLE, ORCHESTRATOR, Y_TAG, MPI_COMM_WORLD);

            // And delete x and y
            delete[] x;
            delete[] y;
        }

    }

    MPI_Finalize();

    return 0;
}

void exact(double x, double y, double* u, double* dudx, double* dudy) {
    double pi = 3.141592653589793;

    *u = sin(pi * x) * sin(pi * y) + x;
    *dudx = pi * cos(pi * x) * sin(pi * y) + 1.0;
    *dudy = pi * sin(pi * x) * cos(pi * y);

    return;
}

double* r8ge_fs_new(int n, double a[], double b[]) {
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
        //
        //  Find the maximum element in column I.
        //
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
        //
        //  Switch rows JCOL and IPIV, and X.
        //
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
        //
        //  Scale the pivot row.
        //
        t = a[jcol - 1 + (jcol - 1) * n];
        a[jcol - 1 + (jcol - 1) * n] = 1.0;
        for (j = jcol + 1; j <= n; j++) {
            a[jcol - 1 + (j - 1) * n] = a[jcol - 1 + (j - 1) * n] / t;
        }
        x[jcol - 1] = x[jcol - 1] / t;
        //
        //  Use the pivot row to eliminate lower entries in that column.
        //
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
    //
    //  Back solve.
    //
    for (jcol = n; 2 <= jcol; jcol--) {
        for (i = 1; i < jcol; i++) {
            x[i - 1] = x[i - 1] - a[i - 1 + (jcol - 1) * n] * x[jcol - 1];
        }
    }

    return x;
}

void timestamp() {
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

int* evenly_divide(int to_be_divided, int parts, int lower_bound) {
    int i = 1;
    int total = 0;
    int partitions = parts + 1;

    int* splits = new int[partitions];

    splits[0] = lower_bound;

    for (; i < to_be_divided % partitions; i++) {
        total += (to_be_divided / parts) + 1;
        splits[i] = total;
    }
    for (; i < partitions; i++) {
        total += (to_be_divided / parts);
        splits[i] = total;
    }

    splits[parts] = to_be_divided;

    return splits;
}