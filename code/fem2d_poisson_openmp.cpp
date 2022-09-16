#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace std;

int main();
void exact(double x, double y, double* u, double* dudx, double* dudy);
double* r8ge_fs_new(int n, double a[], double b[]);
void timestamp();

//****************************************************************************80

int main()

//****************************************************************************80
//
//  Purpose:
//
//    MAIN is the main routine for FEM2D_POISSON_RECTANGLE_LINEAR.
//
//  Discussion:
//
//    This program solves
//
//      - d2U(X,Y)/dx2 - d2U(X,Y)/dy2 = F(X,Y)
//
//    in a rectangular region in the plane.
//
//    Along the boundary of the region, Dirichlet conditions
//    are imposed:
//
//      U(X,Y) = G(X,Y)
//
//    The code uses continuous piecewise linear basis functions on
//    triangles determined by a uniform grid of NX by NY points.
//
//    u    =      sin ( pi * x ) * sin ( pi * y ) + x
//
//    dudx = pi * cos ( pi * x ) * sin ( pi * y ) + 1
//    dudy = pi * sin ( pi * x ) * cos ( pi * y )
//
//    d2udx2 = - pi * pi * sin ( pi * x ) * sin ( pi * y )
//    d2udy2 = - pi * pi * sin ( pi * x ) * sin ( pi * y )
//
//    rhs  = 2 * pi * pi * sin ( pi * x ) * sin ( pi * y )
//
//  THINGS YOU CAN EASILY CHANGE:
//
//    1) Change NX or NY, the number of nodes in the X and Y directions.
//    2) Change XL, XR, YB, YT, the left, right, bottom and top limits of the rectangle.
//    3) Change the exact solution in the EXACT routine, but make sure you also
//       modify the formula for RHS in the assembly portion of the program//
//
//  HARDER TO CHANGE:
//
//    4) Change from "linear" to "quadratic" triangles;
//    5) Change the region from a rectangle to a general triangulated region;
//    6) Store the matrix as a sparse matrix so you can solve bigger systems.
//    7) Handle Neumann boundary conditions.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    28 November 2008
//
//  Author:
//
//    John Burkardt
//
{
    int nx = 5;
    int ny = 5;

    double* a;
    double area;
    double* b;
    double* c;
    double dqidx;
    double dqidy;
    double dqjdx;
    double dqjdy;
    double dudx;
    double dudy;
    int e;
    int* element_node;
    int element_num;
    int i;
    int i1;
    int i2;
    int i3;
    int j;
    int j2;
    int k;
    int node_num;
    int nq1;
    int nq2;
    int nti1;
    int nti2;
    int nti3;
    int ntj1;
    int ntj2;
    int ntj3;
    double pi = 3.141592653589793;
    int q1;
    int q2;
    double qi;
    int ti1;
    int ti2;
    int ti3;
    int tj1;
    int tj2;
    int tj3;
    double rhs;
    double u;
    double wq;
    double* x;
    double xl = 0.0;
    double xq;
    double xr = 1.0;
    double* y;
    double yb = 0.0;
    double yq;
    double yt = 1.0;

    timestamp();
    cout << "\n";
    cout << "FEM2D_POISSON_RECTANGLE_LINEAR\n";
    cout << "  C++ version\n";
    cout << "\n";
    cout << "  Solution of the Poisson equation:\n";
    cout << "\n";
    cout << "  - Uxx - Uyy = F(x,y) inside the region,\n";
    cout << "       U(x,y) = G(x,y) on the boundary of the region.\n";
    cout << "\n";
    cout << "  The region is a rectangle, defined by:\n";
    cout << "\n";
    cout << "  " << xl << " = XL<= X <= XR = " << xr << "\n";
    cout << "  " << yb << " = YB<= Y <= YT = " << yt << "\n";
    cout << "\n";
    cout << "  The finite element method is used, with piecewise\n";
    cout << "  linear basis functions on 3 node triangular\n";
    cout << "  elements.\n";
    cout << "\n";
    cout << "  The corner nodes of the triangles are generated by an\n";
    cout << "  underlying grid whose dimensions are\n";
    cout << "\n";
    cout << "  NX =                       " << nx << "\n";
    cout << "  NY =                       " << ny << "\n";
    //
    //  NODE COORDINATES
    //
    //  Numbering of nodes is suggested by the following 5x10 example:
    //
    //    J=5 | K=41  K=42 ... K=50
    //    ... |
    //    J=2 | K=11  K=12 ... K=20
    //    J=1 | K= 1  K= 2     K=10
    //        +--------------------
    //          I= 1  I= 2 ... I=10
    //
    node_num = nx * ny;

    cout << "  Number of nodes =          " << node_num << "\n";

    x = new double[node_num];
    y = new double[node_num];

    // k = 0;
    // for (j = 1; j <= ny; j++) {
    //     for (i = 1; i <= nx; i++) {
    //         x[k] = ((double)(nx - i) * xl + (double)(i - 1) * xr) / (double)(nx - 1);

    //         y[k] = ((double)(ny - j) * yb + (double)(j - 1) * yt) / (double)(ny - 1);

    //         printf("%d: %d, %d |", k, i, j);

    //         k = k + 1;
    //     }
    // }

    printf("\n");

    {
#pragma omp for
        for (int idx = 0; idx < node_num; idx++) {
            j = 1 + idx / ny;
            i = 1 + idx % ny;

            x[idx] = ((double)(nx - i) * xl + (double)(i - 1) * xr) / (double)(nx - 1);

            y[idx] = ((double)(ny - j) * yb + (double)(j - 1) * yt) / (double)(ny - 1);
        }
    }

    //
    //  ELEMENT array
    //
    //  Organize the nodes into a grid of 3-node triangles.
    //  Here is part of the diagram for a 5x10 example:
    //
    //    |  \ |  \ |  \ |
    //    |   \|   \|   \|
    //   21---22---23---24--
    //    |\ 8 |\10 |\12 |
    //    | \  | \  | \  |
    //    |  \ |  \ |  \ |  \ |
    //    |  7\|  9\| 11\|   \|
    //   11---12---13---14---15---16---17---18---19---20
    //    |\ 2 |\ 4 |\ 6 |\  8|                   |\ 18|
    //    | \  | \  | \  | \  |                   | \  |
    //    |  \ |  \ |  \ |  \ |      ...          |  \ |
    //    |  1\|  3\|  5\| 7 \|                   |17 \|
    //    1----2----3----4----5----6----7----8----9---10
    //
    element_num = 2 * (nx - 1) * (ny - 1);

    cout << "  Number of elements =       " << element_num << "\n";

    element_node = new int[3 * element_num];

    k = 0;

    {
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
    }
    //
    //  Assemble the coefficient matrix A and the right-hand side B of the
    //  finite element equations, ignoring boundary conditions.
    //
    a = new double[node_num * node_num];
    b = new double[node_num];

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
    c = r8ge_fs_new(node_num, a, b);
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

    return 0;
}
//****************************************************************************80

void exact(double x, double y, double* u, double* dudx, double* dudy)

//****************************************************************************80
//
//  Purpose:
//
//    EXACT calculates the exact solution and its first derivatives.
//
//  Discussion:
//
//    The function specified here depends on the problem being
//    solved.  The user must be sure to change both EXACT and RHS
//    or the program will have inconsistent data.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    28 November 2008
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, Y, the coordinates of a point
//    in the region, at which the exact solution is to be evaluated.
//
//    Output, double *U, *DUDX, *DUDY, the value of
//    the exact solution U and its derivatives dUdX
//    and dUdY at the point (X,Y).
//
{
    double pi = 3.141592653589793;

    *u = sin(pi * x) * sin(pi * y) + x;
    *dudx = pi * cos(pi * x) * sin(pi * y) + 1.0;
    *dudy = pi * sin(pi * x) * cos(pi * y);

    return;
}
//****************************************************************************80

double* r8ge_fs_new(int n, double a[], double b[])

//****************************************************************************80
//
//  Purpose:
//
//    R8GE_FS_NEW factors and solves a R8GE system.
//
//  Discussion:
//
//    The R8GE storage format is used for a "general" M by N matrix.
//    A physical storage space is made for each logical entry.  The two
//    dimensional logical array is mapped to a vector, in which storage is
//    by columns.
//
//    R8GE_FS does not save the LU factors of the matrix, and hence cannot
//    be used to efficiently solve multiple linear systems, or even to
//    factor A at one time, and solve a single linear system at a later time.
//
//    R8GE_FS uses partial pivoting, but no pivot vector is required.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    04 December 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the order of the matrix.
//    N must be positive.
//
//    Input/output, double A[N*N].
//    On input, A is the coefficient matrix of the linear system.
//    On output, A is in unit upper triangular form, and
//    represents the U factor of an LU factorization of the
//    original coefficient matrix.
//
//    Input, double B[N], the right hand side of the linear system.
//
//    Output, double R8GE_FS_NEW[N], the solution of the linear system.
//
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
//****************************************************************************80

void timestamp()

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    May 31 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    23 September 2008
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
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
