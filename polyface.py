

""" ID FACEBOOK """
x = [100073378727657, 100026169713124, 100003035062944, 100082912543740, 100005931251080, 100007842199409, 100035542368320, 100074151643179, 100081304663727, 100002668834684, 100003699656193, 100060020855417, 100001529547856, 100011093077016, 100090595661987, 100005674628647, 100008249680556, 100067345147546, 100000729379224, 100001275668122, 100034157994324, 100077975485757]

""" TAHUN PEMBUATAN FACEBOOK """
y = [2021, 2018, 2011, 2022, 2013, 2014, 2019, 2021, 2022, 2011, 2012, 2020, 2011, 2016, 2023, 2013, 2014, 2021, 2010, 2010, 2019, 2022]

""" MIN MAX SCALING X """
min_x = min(x)
max_x = max(x)
x = [(x[i] - min_x) / (max_x - min_x) for i in range(len(x))]

""" MIN MAX SCALING Y """
min_y = min(y)
max_y = max(y)
y = [(y[i] - min_y) / (max_y - min_y) for i in range(len(x))]

titikData = len(x)
m1 = 0
m2 = 0
b = 0
learningRate = 0.01
iterasi = 500000

def train(iterasi=iterasi, learningRate=learningRate):
    global m1, m2, b
    for it in range(iterasi):
        gradientM1 = 0
        gradientM2 = 0
        gradientB = 0
        total_error = 0
        for i in range(titikData):
            y_prediksi = b + m1 * x[i] + m2 * x[i]**2
            error = y[i] - y_prediksi
            total_error += error**2
            gradientM1 += (-2 * x[i] * error)
            gradientM2 += (-2 * x[i]**2 * error)
            gradientB += (-2 * error)
        m1 -= (learningRate * gradientM1) / titikData
        m2 -= (learningRate * gradientM2) / titikData
        b -= (learningRate * gradientB) / titikData
        mse = total_error / titikData
        if it % 1000 == 0:
            print(f"\rMSE (error): {mse}                                             ")
        print("\rProses training sedang dilakukan, lama proses tergantung pada jumlah iterasi", end="")
    print(f"\rParameter m1, m2, dan b setelah training: ({m1}, {m2}, {b})")

def prediksi(x_test):
    hasilPrediksi = b + m1 * x_test + m2 * x_test**2
    return minMaxScalingReverserY(hasilPrediksi)

def minMaxScalingX(x_test):
    return (x_test - min_x) / (max_x - min_x)

def minMaxScalingReverserY(y_scaled):
    return (y_scaled * (max_y - min_y)) + min_y

if __name__ == "__main__":
    try:
        listParameter = eval(open("parameter.txt", "r").read())
        print("Pilih y untuk menggunakan parameter yang sudah ditrain (REKOMENDED)")
        print("Pilih enter untuk melakukan training ulang (jika ingin mengubah learning rate, iterasi, atau data latih)")
        par = input("Gunakan parameter yang sudah ditrain sebelumnya (y/enter): ").lower()
        if par == "y":
            m1, m2, b = listParameter
        else:
            iterasi = int(input("Masukkan jumlah iterasi (rekomendasi antara 50000 sampai 500000): "))
            lr = float(input("Masukkan learning rate (rekomendasi antara 0.01 sampai 0.0001): "))
            train(iterasi=iterasi, learningRate=lr)
            open("parameter.txt", "w").write(f"[{m1}, {m2}, {b}]")
    except:
        train()
        open("parameter.txt", "w").write(f"[{m1}, {m2}, {b}]")
    testX1 = x[0]
    testX2 = x[1]
    testX3 = x[2]
    tahunAsliX1 = minMaxScalingReverserY(y[0])
    tahunAsliX2 = minMaxScalingReverserY(y[1])
    tahunAsliX3 = minMaxScalingReverserY(y[2])
    tahunPrediksiX1 = prediksi(testX1)
    tahunPrediksiX2 = prediksi(testX2)
    tahunPrediksiX3 = prediksi(testX3)
    print("\n"*2)
    print("Membandingkan 3 data x pertama dengan nilai y sebenarnya")
    print(f"Prediksi tahun x[0]: {tahunPrediksiX1:.0f}, Tahun aslinya: {tahunAsliX1:.0f}")
    print(f"Prediksi tahun x[1]: {tahunPrediksiX2:.0f}, Tahun aslinya: {tahunAsliX2:.0f}")
    print(f"Prediksi tahun x[2]: {tahunPrediksiX3:.0f}, Tahun aslinya: {tahunAsliX3:.0f}")
    print("\n"*2)
    while True:
        idFacebook = int(input("Masukkan id: "))
        if len(str(idFacebook)) != 15:
            print("Maaf id seperti ini belum disupport")
            continue
        prediksiTahunPembuatanFacebook = prediksi(minMaxScalingX(idFacebook))
        print(f"Prediksi tahun pembuatan facebook dengan id {idFacebook}: {prediksiTahunPembuatanFacebook:.0f}")
