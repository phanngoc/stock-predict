# Stock predic tendency
Đây là repository tập hợp các notebook  mình tìm hiểu về chủ đề dựa vào text và các bài viết 
(bằng tiếng việt) để dự đoán khuynh hướng giá chỉ số VN-Index
.Các code + kiến thức chỉ mang tính học tập. Mọi người sử dụng một cách cẩn thận và có trách nhiệm với mục đích của mình nhé :)

- rnn-stock-vnindex : dự đoán chỉ dựa vào chỉ số, dùng RNN + LSTM
- w2vector-predict-stock-tendency.ipynb: Dùng Word2vec + RNN để dự đoán khuynh hướng, data chỉ mang mục đích sample.
- w2vector-linear-predict-stock-tendency: phobert + RNN, mình thấy cũng giống cái trên, nhưng data lấy từ model "vinai/phobert-base-v2" dùng lib tranformer, data ít nên mục đích chỉ để adapt thuật toán :v.
- w2vector-linear-predict-stock-tendency : word2vec + SVR, dự đoán chỉ yếu dựa trên SVR + kernel linear, data được cắt tỉa dựa vào trung bình cộng của array từ model word2vec, nói chung cũng khá nhẹ nhàng :v.

