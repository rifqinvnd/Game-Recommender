# Game-Recommender

# Recommendation System with *Content-based Filtering* *Oleh: [Rifqi Novandi](https://github.com/rifqinvnd)* 

## Latar Belakang
Game atau permainan merupakan sarana yang digunakan untuk bermain, sebuah barang atau sesuatu yang pada umumnya digunakan untuk hiburan atau kesenangan, dan kadang-kadang digunakan sebagai alat pendidikan. Permainan berbeda dari pekerjaan, yang biasanya dilakukan untuk mendapatkan upah, dan dari seni, yang lebih sering merupakan ekspresi elemen estetika atau ideologis. Namun, perbedaannya tidak jelas, dan banyak permainan juga dianggap sebagai karya (seperti pemain profesional olahraga atau permainan penonton) atau seni (seperti puzzle atau permainan yang melibatkan tata letak artistik seperti Mahjong, solitaire, atau beberapa permainan video).  

Game sangat relevan dengan kondisi saat ini yaitu pandemi dimana kebanyakan orang menggunakan game sebagai sarana untuk menghilangkan rasa jenuh di rumah. Terdapat banyak genre dari game seperti action, sport, music, dll. Game juga mempunyai platform tertentu seperti yang dimainkan di PC, PS, Handphone, dll. Biasanya orang dengan genre game tertentu seperti action, cenderung memainkan game dengan genre action juga. Begitu pula dengan platform, orang yang biasa memainkan game pada platform tertentu semisal PC, cenderung memainkan game yang berada di platform yang sama.  

Pada proyek machine learning ini, akan dibuat model sistem rekomendasi untuk memprediksi game yang disukai berdasarkan game lain yang memiliki kesamaan serupa atau dengan menggunakan teknik *content-based filtering* dengan beberapa variabel seperti platform, tahun rilis, genre, dll.  

Seperti yang dijelaskan sebelumnya, orang yang memainkan genre game tertentu diplatform tertentu cenderung stuck dengan genre dan platform yang sama. Ataupun karena mereka puas dengan game yang dilaunch publisher tertentu, maka akan menantikan game yang juga dilaunch oleh publisher tersebut. Oleh karena itu, disini akan dibuat sistem rekomendasi game yang dapat memprediksi game-game yang mungkin disukai oleh pengguna berdasarkan genre, platform, publisher, dan variabel lain yang memiliki kesamaan atau dengan *content-based filtering*.   

*Referensi* : [Wikipedia](https://id.wikipedia.org/wiki/Permainan#Jenis_permainan)  

## Business Understanding 
Untuk menciptakan sistem rekomendasi yang dapat merekomendasikan game yang mugkin disukai pengguna, akan digunakan teknik model sistem rekomendasi *content-based filtering* dengan algoritma KNearestNeighbors dan juga CosineSimilarity.  

- Masalah yang harus diselesaikan: 
Mendapatkan rekomendasi game tertentu yang mungkin disukai oleh pengguna 
- Tujuan dari pembuatan Machine Learning Model: 
Model dapat merekomendasikan game tertentu dengan tingkat kesamaan yang tinggi (>90%). 
- Solusi: 
Membuat Machine Learning dengan Algoritma KNearestNeighbors yang dapat merekomendasikan game berdasarkan k-kemiripan fitur game tertentu.  

## Data Understanding 
Dataset yang digunakan untuk merekomendasikan game merupakan dataset dari Kaggle. Dataset ini telah digunakan untuk memprediksi stroke dengan 242 algoritma model berbeda. Dataset ini memiliki 16719 sample atau row dengan 16 fitur atau kolom untuk membuat sistem rekomendasi game dengan *content-based filtering*. Dataset ini dibuat oleh [Rush Kirubi](https://www.kaggle.com/rush4ratio).  

Link Dataset: [Game Sales with Ratings Dataset](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings)  

Feature pada Dataset: 
1. Name : Nama game yang akan direkomendasikan [str] 
2. Platform : Platform tempat game disediakan (terdapat 31 platform berbeda) [str] 
3. Year_of_Release : Tahun rilis game [str] 
4. Genre : Genre atau jenis game yang berisikan 12 genre berbeda [str] 
5. Publisher : Penerbit yang menerbitkan game [str] 
6. NA_Sales : Sales game pada region North America dalam satuan juta [float] 
7. EU_Sales : Sales game pada region Europe dalam satuan juta [float] 
8. JP_Sales : Sales game pada region Japan dalam satuan juta [float] 
9. Other_Sales : Sales game pada region selain North America, Europe dan Japan dalam satuan juta [float] 
10. Global_Sales : Total sales game pada seluruh region dalam satuan juta [float] 
11. Critic_Score : Nilai agregat yang diberikan oleh metacritic reviewer [float] 
12. Critic_Count : Total metacritic reviewer yang menilai game [float] 
13. User_Score : Nilai agregat yang diberikan oleh user [float] 
14. User_Count : Total user yang menilai game [float] 
15. Developer : Perusahaan yang berkolaborasi dalam pembuatan game [str] 
16. Rating : ESRB Rating menyediakan rating yang biasanya mendeskripsikan kalangan tertentu dalam rentang umur tertentu ('E', 'M', 'T', 'E10+', 'K-A', 'AO', 'EC', 'RP') [str]  

## Data Preparation 
Untuk tahap persiapan data, telah dilakukan beberapa tahap yaitu dengan mmembuang data kosong pada setiap kolom, menghilangkan kolom yang tidak diperlukan seperti kolom Critic_Score, Critic_Count,dan User_Count serta kolom Global_Sales, membuang unique elemen pada kolom dengan unique elemen yang bernilai sedikit seperti kolom Platform, membuang data duplikasi, melakukan one-hot encoding pada data kategorikal, serta melakukan standarisasi pada kolom numerikal dengan MinMaxScaler.  

Diperlukan tahapan seperti membuang data kosong agar perhitungan atau algoritma tidak error. Untuk tahapan membuang kolom tertentu seperti kolom Critic_Score, Critic_Count,dan User_Count karena terlalu banyak data kosong serta kolom Global_Sales karena telah dideskripsikan oleh sales pada region yang telah dibagi diperlukan agar data data yang dapat merusak model dapat dibuang. Setelah itu, membuang unique elemen pada kolom dengan unique elemen yang bernilai sedikit seperti kolom Platform dengan jumlah dibawah 350 agar kualitas data dapat meningkat. Lalu membuang data duplikasi agar kualitas model juga baik. Untuk tahapan standarisasi dilakukan agar fitur tidak terbanting nilainya dengan fitur lainnya menggunakan MinMaxScaler serta one-hot encoding agar membuat fitur kategorikal menjadi numerikal.  

## Modeling 
Seperti yang sudah dijelaskan diawal, pemodelan machine learning untuk merekomendasikan game tertentu pada pengguna yaitu menggunakan algoritma K-NearestNeighbors. Algoritma ini bekerja berdasarkan dengan fitur fitur yang ada dan kemiripan antara fitur fitur tersebut untuk menilai apakah game tertentu dapat direkomendasikan saat pengguna memainkan sebuah game.  

Dengan data yang ada dan setelah dilakukan pengolahan data, diambil fitur fitur yang berpengaruh  terhadap rekomendasi game. Beberapa fitur yang digunakan yaitu Platform, Year_of_Release, Game, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, User_Score, dan Rating.  

Pada awal pembuatan model, digunakan model K-NearestNeighbors dengan set parameter metric yang digunakan yaitu euclidean distance dan dibuat fungsi sistem rekomendasi untuk 5 game teratas saat sebuah game diberikan. Dibuat sebuah list dari nama game yang direkomendasikan serta kemiripannya dimana 100% dikurang dengan euclidean distance dari game yag direkomendasikan. Lalu list tersebut dimasukkan pada DataFrame sehingga dapat mudah dipahami oleh pengguna.  

Menggunakan model awal, dilakukan prediksi pada game dengan nama Final Fantasy IX atau loc[111] pada DataFrame nama game. Didapatkan hasil berupa rekomendasi game seperti Final Fantasy VIII, Final Fantasy Tactics, Xenogears, Tales of Destiny II, dan Chrono Cross yang semua game tersebut memiliki kemiripan berdasarkan euclidean distance diatas 98,5%. Rekomendasi yang cukup baik untuk model awal dengan beberapa game serupa ditunjukan dengan nama dari game rekomendasi yang sama dengan game yang dimainkan.  

Dilakukan pengembangan machine learning dengan menggunakan Algoritma yang berbeda yaitu dimana menggunakan cosine similarity. Setelah dilakukan pembuatan dataframe cosinesimilarity dan pembuatan fungsi rekomendasi dengan cosine similarity, sama seperti model awal, hasil dari rekomendasi game dimasukkan pada suatu dataframe yang berisikan nama game rekomendasi serta kemiripannya menggunakan cosinesimilarity.   

Ternyata dengan menggunakan Algoritma yang berbeda, hasil dari prediksi game Final Fantasy IX atau loc[111] pada DataFrame nama game sama dengan model awal menggunakan KNearestNeighbors. Meskipun rekomendasi game seperti Final Fantasy VIII, Final Fantasy Tactics, Xenogears, Tales of Destiny II, dan Chrono Cross memiliki score cosinesimilarity diatas 0,82 , namun dapat disimpulkan bahwa model sukses memprediksi game yang mungkin disukai pengguna.  

## Evaluasi Model 
Untuk evaluasi model, digunakan 2 metode atau cara evaluasi model yaitu dengan Calinski-Harabasz Score dan Davies-Bouldin Score. 

Calinski-Harabasz atau juga dikenal sebagai Variance Ratio Criterion, merupakan rasio jumlah dispersi antar-cluster dan dispersi antar-cluster untuk semua cluster, semakin tinggi skornya, semakin baik kinerjanya. Didapatkan score 5.09 untuk Calinski-Harabasz score yang dimana cukup kecil untuk model sistem rekomendasi.  

Metode yang kedua yaitu dengan menggunakan evaluasi Davies-Bouldin Score. Score ini menandakan rata-rata ‘kemiripan’ antar klaster, dimana kemiripan adalah ukuran yang membandingkan jarak antar klaster dengan ukuran klaster itu sendiri. Davies-Bouldin Score yang lebih rendah berhubungan dengan model yang memiliki pemisahan yang lebih baik antara cluster. Didapatkan score 2.93 untuk Davies-Bouldin score yang dimana cukup tinggi untuk model sistem rekomendasi.  

### Referensi 
- Dokumentasi Scikit-learn: [https://scikit-learn.org/stable/modules/classes.html](https://scikit-learn.org/stable/modules/classes.html) 
- Referensi Laporan: [https://github.com/fahmij8/ML-Exercise/blob/main/MLT-2/MLT_Proyek_Submission_2.ipynb](https://github.com/fahmij8/ML-Exercise/blob/main/MLT-2/MLT_Proyek_Submission_2.ipynb) 
- Wikipedia: [Permainan (Game)](https://id.wikipedia.org/wiki/Permainan#Jenis_permainan) 
- Dataset: [Game Sales with Ratings Dataset](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings)
