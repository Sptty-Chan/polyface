
cdef:
    long int[22] xm = [100073378727657, 100026169713124, 100003035062944, 100082912543740, 100005931251080, 100007842199409, 100035542368320, 100074151643179, 100081304663727, 100002668834684, 100003699656193, 100060020855417, 100001529547856, 100011093077016, 100090595661987, 100005674628647, 100008249680556, 100067345147546, 100000729379224, 100001275668122, 100034157994324, 100077975485757]

    int[22] ym = [2021, 2018, 2011, 2022, 2013, 2014, 2019, 2021, 2022, 2011, 2012, 2020, 2011, 2016, 2023, 2013, 2014, 2021, 2010, 2010, 2019, 2022]

    long int min_x = min(xm)
    long int max_x = max(xm)

    int i
    int titikData = len(xm)
    double[22] x = [(xm[i] - min_x) / (max_x - min_x) for i in range(titikData)]

    int min_y = min(ym)
    int max_y = max(ym)
    double[22] y = [(ym[i] - min_y) / (max_y - min_y) for i in range(titikData)]

    double m1 = 0
    double m2 = 0
    double m3 = 0
    double b = 0
    double learningRate = 0.05
    int iterasi = 500000

cpdef void train(int iterasi=iterasi, double learningRate=learningRate):
    global m1, m2, m3, b
    cdef:
        int i
        int it
        double gradientM1 = 0
        double gradientM2 = 0
        double gradientM3 = 0
        double gradientB = 0
        double total_error = 0
        double y_prediksi
        double error
        double mse
        str inf = "\rProses training sedang dilakukan, lama proses tergantung pada jumlah iterasi"
    for it in range(iterasi):
        gradientM1 = 0
        gradientM2 = 0
        gradientM3 = 0
        gradientB = 0
        total_error = 0
        for i in range(titikData):
            y_prediksi = b + m1 * x[i] + m2 * x[i]**2 + m3 * x[i]**3
            error = y_prediksi - y[i]
            total_error += error**2
            gradientM1 += (2 * x[i] * error)
            gradientM2 += (2 * x[i]**2 * error)
            gradientM3 += (2 * x[i]**3 * error)
            gradientB += (2 * error)
        m1 -= (learningRate * gradientM1) / titikData
        m2 -= (learningRate * gradientM2) / titikData
        m3 -= (learningRate * gradientM3) / titikData
        b -= (learningRate * gradientB) / titikData
        mse = total_error / titikData
        if it % 10000 == 0:
            print(f"\rMSE (error): {mse}                                             ")
            print(inf, end="")
    print(f"\rParameter m1, m2, m3, dan b setelah training: ({m1}, {m2}, {m3}, {b})")

cpdef int prediksi(double x_test, double m1, double m2, double m3, double b):
    cdef double hasilPrediksi = b + m1 * x_test + m2 * x_test**2 + m3 * x_test**3
    return minMaxScalingReverserY(hasilPrediksi)

cpdef double minMaxScalingX(long int x_test):
    cdef double scaled = (x_test - min_x) / (max_x - min_x)
    return scaled

cpdef minMaxScalingReverserY(double y_scaled):
    cdef double reversedY = (y_scaled * (max_y - min_y)) + min_y
    return reversedY

cpdef (double, double) getDataSet(int index):
    return (x[index], y[index])

cpdef (double, double, double, double) getParameter():
    return (m1, m2, m3, b)
