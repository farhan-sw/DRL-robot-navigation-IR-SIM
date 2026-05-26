# Panduan DRL Robot Navigation + A* Global Planner

Repositori ini telah dimodifikasi untuk mendukung arsitektur navigasi hibrida (Hybrid Navigation Architecture), di mana **Algoritma A*** bertugas sebagai *Global Path Planner* (mencari rute dari awal ke akhir), dan agen **Reinforcement Learning (CNN-TD3)** bertugas sebagai *Local Path Planner* (mengikuti jalur rute A* secara dinamis sambil menghindari rintangan).

Berikut adalah panduan lengkap cara melatih (*training*) dan menguji (*testing*) sistem ini.

---

## 1. Pelatihan Agen RL (Training)

Sebelum agen RL bisa digunakan untuk menavigasi rute A*, agen ini harus dilatih terlebih dahulu melalui skrip `rl_train.py` agar "pintar" mengendarai robot dan menghindari tabrakan.

### A. Training Mode Sangat Cepat (Tanpa GUI)
Untuk melatih agen dengan kecepatan maksimal (ribuan kali lebih cepat), jalankan program tanpa parameter tambahan. Layar GUI simulasi IRSim akan dimatikan secara otomatis sehingga CPU bisa fokus 100% pada kalkulasi matriks *Neural Network*.
```bash
poetry run python robot_nav/rl_train.py
```

### B. Menonton Proses Training (Dengan GUI)
Jika Anda ingin melihat secara visual bagaimana robot bergerak mengeksplorasi arena selama proses *training*, tambahkan *flag* `--render`.
*(Catatan: Menggunakan parameter ini akan memperlambat proses training secara amat sangat drastis karena beban rendering grafis).*
```bash
poetry run python robot_nav/rl_train.py --render
```

### C. Training Lebih Cepat (Fast Train)
Jika Anda merasa proses perhitungan kalkulasi masih terlalu memakan waktu, Anda bisa menggunakan *flag* `--fast-train`. Opsi ini akan memotong beban iterasi *Neural Network* sebesar 50% setiap episodenya. Kekurangannya, robot mungkin membutuhkan waktu eksplorasi lebih lama (lebih banyak ronde/siklus) untuk bisa sepintar versi aslinya.
```bash
poetry run python robot_nav/rl_train.py --fast-train
```

### D. Melanjutkan Training (Auto-Resume)
Jika proses training sebelumnya terhenti (misalnya laptop mati atau dihentikan manual), Anda **tidak perlu mengajari robot dari nol (bodoh) lagi**. Gunakan *flag* `--resume` untuk otomatis memuat bobot (*weights*) parameter terakhir.
```bash
poetry run python robot_nav/rl_train.py --resume
```
*(Tips: Parameter bisa digabung, contohnya: `poetry run python robot_nav/rl_train.py --resume --render`)*

### E. Fitur Auto-Save & Safe Exit (Ctrl + C)
- **Auto-Save Rutin**: Sistem akan menyimpan model secara otomatis setiap 5 siklus pelatihan ke dalam folder `robot_nav/models/CNNTD3/checkpoint/`.
- **Safe Exit (Penyelamat Data)**: Jika sewaktu-waktu Anda menekan `Ctrl + C` di terminal untuk menghentikan program, sistem tidak akan langsung *crash*, melainkan mencegat instruksi tersebut, **menyimpan posisi otak model saat itu juga**, lalu keluar dengan aman.

---

## 2. Pengujian Arsitektur Hibrida (A* + RL)

Setelah Anda membiarkan proses *training* berjalan cukup lama dan robot sudah cukup mahir, saatnya menguji sistem navigasi hibridanya menggunakan skrip `rl_a_star_test.py`.

Proses yang terjadi pada skrip pengujian ini:
1. Membangun Peta (*Occupancy Grid*) secara instan.
2. Mencari jalur A* dari robot menuju target akhir (ditandai dengan **Garis Putus-Putus Merah** / *Global Path*).
3. Mengekstrak titik-titik *waypoints* (ditandai dengan **Titik-Titik Biru** / *Local Goals*).
4. Robot RL akan menyetir secara dinamis mengejar titik-titik biru tersebut satu per satu.

### Cara Menjalankan Pengujian:
```bash
poetry run python robot_nav/rl_a_star_test.py
```

**⚠️ PERHATIAN PENTING SEBELUM TESTING:**
Secara *default*, skrip pengujian saat ini **sudah diatur untuk otomatis memuat (load) hasil pelatihan (weights)** dari folder `checkpoint`. Jadi Anda tidak perlu menambahkan argumen apa-apa jika sudah pernah melakukan training.

Jika Anda **belum memiliki file hasil training** dan hanya ingin melihat garis A* dengan robot yang bergerak acak (karena otaknya kosong), Anda harus secara eksplisit mematikan fitur *load model* dengan argumen `--random-weights`:

```bash
poetry run python robot_nav/rl_a_star_test.py --random-weights
```
