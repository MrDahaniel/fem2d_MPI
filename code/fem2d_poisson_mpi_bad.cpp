#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

MPI_Win a_window, b_window, c_window;

int* evenly_divide(int to_be_divided, int parts, int lower_bound = 0) {
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

void exact(double x, double y, double* u, double* dudx, double* dudy) {
    double pi = 3.141592653589793;

    *u = sin(pi * x) * sin(pi * y) + x;
    *dudx = pi * cos(pi * x) * sin(pi * y) + 1.0;
    *dudy = pi * sin(pi * x) * cos(pi * y);

    return;
}

void r8ge_fs_new(int n, int lower, int upper, double*& a, double*& b, double*& x) {
    int i;
    int ipiv, r_ipiv;
    int j;
    int jcol, l_jcol;
    double piv, r_piv;
    double t;

    int* splits;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (jcol = 1; jcol <= n; jcol++) {
        r_piv = fabs(a[jcol - 1 + (jcol - 1) * n]);
        r_ipiv = jcol;

        for (i = lower; i <= upper; i++) {
            if (r_piv < fabs(a[i - 1 + (jcol - 1) * n])) {
                r_piv = fabs(a[i - 1 + (jcol - 1) * n]);
                r_ipiv = i;
            }
        }

        // Reduce + broadcast of the pivot
        MPI_Allreduce(&r_piv, &piv, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&r_ipiv, &ipiv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (piv == 0.0) {
            std::cout << "\n";
            std::cout << "R8GE_FS_NEW - Fatal error!\n";
            printf("rank %d lower %d upper %d\n", rank, lower, upper);
            std::cout << "Rank " << rank << " Zero pivot on step " << jcol << "\n";
            return;
        }

        if (jcol != ipiv) {
            MPI_Win_fence(0, a_window);

            for (j = lower; j <= upper; j++) {
                t = a[jcol - 1 + (j - 1) * n];
                a[jcol - 1 + (j - 1) * n] = a[ipiv - 1 + (j - 1) * n];
                a[ipiv - 1 + (j - 1) * n] = t;
            }

            MPI_Win_fence(0, a_window);
            MPI_Win_fence(0, c_window);

            if (rank == 0) {
                t = x[jcol - 1];
                x[jcol - 1] = x[ipiv - 1];
                x[ipiv - 1] = t;
            }

            MPI_Win_fence(0, c_window);
        }

        MPI_Win_fence(0, a_window);
        MPI_Win_fence(0, c_window);

        if (rank == 0) {
            t = a[jcol - 1 + (jcol - 1) * n];
            a[jcol - 1 + (jcol - 1) * n] = 1.0;

            for (j = jcol + 1; j <= n; j++) {
                a[jcol - 1 + (j - 1) * n] = a[jcol - 1 + (j - 1) * n] / t;
            }

            x[jcol - 1] = x[jcol - 1] / t;
        }

        MPI_Win_fence(0, a_window);
        MPI_Win_fence(0, c_window);
        // if (rank == 0) {

        MPI_Barrier(MPI_COMM_WORLD);

        for (i = jcol + 1; i <= n; i++) {
            MPI_Win_sync(a_window);
            MPI_Win_sync(c_window);
            if (a[i - 1 + (jcol - 1) * n] != 0.0) {
                MPI_Win_sync(a_window);
                t = -a[i - 1 + (jcol - 1) * n];
                a[i - 1 + (jcol - 1) * n] = 0.0;
                MPI_Win_sync(a_window);

                for (j = lower + 1; j <= upper - 1; j++) {
                    a[i - 1 + (j - 1) * n] += t * a[jcol - 1 + (j - 1) * n];
                }

                MPI_Win_sync(a_window);
                MPI_Win_sync(c_window);

                x[i - 1] = x[i - 1] + t * x[jcol - 1];
                MPI_Win_sync(c_window);
            }
            MPI_Win_sync(a_window);
            MPI_Win_sync(c_window);
        }

        MPI_Win_fence(0, a_window);
        MPI_Win_fence(0, c_window);
    }

    if (rank == 0) {
        for (jcol = n; 2 <= jcol; jcol--) {
            for (i = 1; i < jcol; i++) {
                x[i - 1] = x[i - 1] - a[i - 1 + (jcol - 1) * n] * x[jcol - 1];
            }
        }
    }
}

void timestamp() {
#define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm* tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    std::cout << time_buffer << "\n";

    return;
#undef TIME_SIZE
}

int main(void) {
    // MPI init
    MPI_Init(NULL, NULL);

    MPI_Aint winsize;
    int windisp;

    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int nx = 150;
    int ny = 150;

    double *a, *local_a;
    double *b, *local_b;
    double *c, *local_c;

    double area;

    double dqidx, dqidy, dqjdx, dqjdy, dudx, dudy;

    int e;
    int* element_node;

    int node_num = nx * ny;
    int element_num = 2 * (nx - 1) * (ny - 1);

    // Initialise shared memory
    int local_matrix_size = 0;
    int local_vector_size = 0;

    if (rank == 0) {
        local_matrix_size = node_num * node_num;
        local_vector_size = node_num;
    }

    // And allocate those mf
    MPI_Win_allocate_shared(local_matrix_size * sizeof(double), sizeof(double),
                            MPI_INFO_NULL, MPI_COMM_WORLD, &local_a, &a_window);
    MPI_Win_allocate_shared(local_vector_size * sizeof(double), sizeof(double),
                            MPI_INFO_NULL, MPI_COMM_WORLD, &local_b, &b_window);
    MPI_Win_allocate_shared(local_vector_size * sizeof(double), sizeof(double),
                            MPI_INFO_NULL, MPI_COMM_WORLD, &local_c, &c_window);

    int flag;
    int* model;

    MPI_Win_get_attr(a_window, MPI_WIN_MODEL, &model, &flag);

    if (1 != flag) {
        printf("Attribute MPI_WIN_MODEL not defined\n");
    } else {
        if (MPI_WIN_UNIFIED == *model) {
            if (rank == 0) printf("Memory model is MPI_WIN_UNIFIED\n");
        } else {
            if (rank == 0) printf("Memory model is *not* MPI_WIN_UNIFIED\n");

            MPI_Finalize();
            return 1;
        }
    }

    a = local_a;
    b = local_b;
    c = local_c;

    if (rank != 0) {
        MPI_Win_shared_query(a_window, 0, &winsize, &windisp, &a);
        MPI_Win_shared_query(b_window, 0, &winsize, &windisp, &b);
        MPI_Win_shared_query(c_window, 0, &winsize, &windisp, &c);
    }

    int i, i1, i2, i3, j, j2, k, nq1, nq2, nti1, nti2, nti3, ntj1, ntj2, ntj3;

    int q1, q2;

    double qi;

    int ti1, ti2, ti3, tj1, tj2, tj3;

    double rhs, u;

    double wq, xq, yq;

    double* x;
    double* y;

    const double pi = 3.141592653589793;
    const double xl = 0.0;
    const double xr = 1.0;
    const double yb = 0.0;
    const double yt = 1.0;

    MPI_Win_fence(0, a_window);
    MPI_Win_fence(0, b_window);
    MPI_Win_fence(0, c_window);

    if (rank == 0) {
        timestamp();
        // std::cout << "\n";
        // std::cout << "FEM2D_POISSON_RECTANGLE_LINEAR\n";
        // std::cout << "  C++ version\n";
        // std::cout << "\n";
        // std::cout << "  Solution of the Poisson equation:\n";
        // std::cout << "\n";
        // std::cout << "  - Uxx - Uyy = F(x,y) inside the region,\n";
        // std::cout << "       U(x,y) = G(x,y) on the boundary of the region.\n";
        // std::cout << "\n";
        // std::cout << "  The region is a rectangle, defined by:\n";
        // std::cout << "\n";
        // std::cout << "  " << xl << " = XL<= X <= XR = " << xr << "\n";
        // std::cout << "  " << yb << " = YB<= Y <= YT = " << yt << "\n";
        // std::cout << "\n";
        // std::cout << "  The finite element method is used, with piecewise\n";
        // std::cout << "  linear basis functions on 3 node triangular\n";
        // std::cout << "  elements.\n";
        // std::cout << "\n";
        // std::cout << "  The corner nodes of the triangles are generated by an\n";
        // std::cout << "  underlying grid whose dimensions are\n";
        // std::cout << "\n";
        // std::cout << "  NX =                       " << nx << "\n";
        // std::cout << "  NY =                       " << ny << "\n";
        // std::cout << "  Number of nodes =          " << node_num << "\n";
        // std::cout << "  Number of elements =       " << element_num << "\n";

        x = new double[node_num];
        y = new double[node_num];

        for (int idx = 0; idx < node_num; idx++) {
            j = 1 + idx / ny;
            i = 1 + idx % ny;

            x[idx] = ((double)(nx - i) * xl + (double)(i - 1) * xr) / (double)(nx - 1);
            y[idx] = ((double)(ny - j) * yb + (double)(j - 1) * yt) / (double)(ny - 1);
        }

        element_node = new int[3 * element_num];

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

        // a = new double[node_num * node_num];
        // b = new double[node_num];

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

        for (int i = 0; i < node_num; i++) {
            c[i] = b[i];
        }
    }

    MPI_Win_fence(0, a_window);
    MPI_Win_fence(0, b_window);
    MPI_Win_fence(0, c_window);

    int* splits = evenly_divide(node_num, size);

    // add 1 to avoid overlaps
    splits[rank] += 1;

    printf("rank %d l: %d u: %d\n", rank, splits[rank], splits[rank + 1]);

    MPI_Barrier(MPI_COMM_WORLD);

    r8ge_fs_new(node_num, splits[rank], splits[rank + 1], a, b, c);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "  Rank   K     I     J          X           Y        U               U                Error\n";
        std::cout << "                                                 exact           computed\n";
        std::cout << "\n";

        k = 0;

        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                exact(x[k], y[k], &u, &dudx, &dudy);

                std::cout << "  " << std::setw(4) << rank
                          << "  " << std::setw(4) << k
                          << "  " << std::setw(4) << i
                          << "  " << std::setw(4) << j
                          << "  " << std::setw(10) << x[k]
                          << "  " << std::setw(10) << y[k]
                          << "  " << std::setw(14) << u
                          << "  " << std::setw(14) << c[k]
                          << "  " << std::setw(14) << fabs(u - c[k]) << "\n";

                k = k + 1;
            }
            std::cout << "\n";
        }

        // delete[] a;
        // delete[] b;
        // delete[] c;
        delete[] element_node;
        delete[] x;
        delete[] y;

        std::cout << "\n";
        std::cout << "FEM2D_POISSON_RECTANGLE_LINEAR:\n";
        std::cout << "  Normal end of execution.\n";
        std::cout << "\n";
        timestamp();
    }

    MPI_Finalize();

    return 0;
}