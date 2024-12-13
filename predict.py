try:
    from polyface import train, minMaxScalingX, minMaxScalingReverserY, prediksi, getDataSet, getParameter
except ModuleNotFoundError:
    print("!!. Sedang melakukan konfigurasi awal. jangan ditutup atau diberhentikan, proses ini hanya dilakukan sekali saat pertama kali program dijalankan.\n\n")
    import os
    os.system("python setup.py build_ext --inplace")
    print("\n\n✓✓. Proses konfigurasi selesai, jalankan ulang program untuk memulai")
    exit()

if __name__ == "__main__":
    try:
        listParameter = eval(open("parameter.txt", "r").read())
        print("Pilih y untuk menggunakan parameter yang sudah ditrain (REKOMENDED)")
        print("Pilih enter untuk melakukan training ulang (jika ingin mengubah learning rate, iterasi, atau data latih)")
        par = input("Gunakan parameter yang sudah ditrain sebelumnya (y/enter): ").lower()
        if par == "y":
            m1, m2, m3, b = listParameter
        else:
            iterasi = int(input("Masukkan jumlah iterasi (rekomendasi antara 50000 sampai 500000): "))
            lr = float(input("Masukkan learning rate (rekomendasi antara 0.01 sampai 0.0001): "))
            train(iterasi=iterasi, learningRate=lr)
            m1, m2, m3, b = getParameter()
            open("parameter.txt", "w").write(f"[{m1}, {m2}, {m3}, {b}]")
    except:
        train()
        m1, m2, m3, b = getParameter()
        open("parameter.txt", "w").write(f"[{m1}, {m2}, {m3}, {b}]")
    testX1, yact1 = getDataSet(0)
    testX2, yact2 = getDataSet(1)
    testX3, yact3 = getDataSet(2)
    tahunAsliX1 = minMaxScalingReverserY(yact1)
    tahunAsliX2 = minMaxScalingReverserY(yact2)
    tahunAsliX3 = minMaxScalingReverserY(yact3)
    tahunPrediksiX1 = prediksi(testX1, m1, m2, m3, b)
    tahunPrediksiX2 = prediksi(testX2, m1, m2, m3, b)
    tahunPrediksiX3 = prediksi(testX3, m1, m2, m3, b)
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
        prediksiTahunPembuatanFacebook = prediksi(minMaxScalingX(idFacebook), m1, m2, m3, b)
        print(f"Prediksi tahun pembuatan facebook dengan id {idFacebook}: {prediksiTahunPembuatanFacebook:.0f}")
