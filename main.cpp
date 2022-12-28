#include "model.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>


Tensor2 read_csv(const std::string &filename,
                 bool one_hot,
                 std::size_t attr_size,
                 std::size_t data_size)
{
    Tensor2 result(attr_size, data_size);
    std::ios_base::sync_with_stdio(false);
    std::ifstream fin(filename.c_str());
    std::string line;
    std::stringstream ss;
    std::size_t r = 0;

    if (!one_hot)
        while (std::getline(fin, line) && r < data_size)
        {
            std::istringstream iss(line);
            std::string temp;
            std::size_t i = 0;

            while (std::getline(iss, temp, ','))
            {
                ss << temp, ss >> result[i++][r];
                ss.str(""), ss.clear();
            }

            ++r;
        }
    else
        while (std::getline(fin, line) && r < data_size)
        {
            std::size_t val;
            ss << line, ss >> val;

            for (std::size_t i = 0; i < attr_size; ++i)
                result[i][r] = 0;

            result[val][r++] = 1;
            ss.str(""), ss.clear();
        }

    return result;
}

std::vector<Tensor2> divide(const Tensor2 &whole,
                            std::size_t batch_size)
{
    std::vector<Tensor2> result;
    Tensor2 batch(whole.get_len1(), batch_size);
    std::size_t quo = whole.get_len2() -
                      whole.get_len2() % batch_size;

    for (std::size_t n = 0; n < quo; n += batch_size)
    {
        for (std::size_t i = 0; i < whole.get_len1(); ++i)
            for (std::size_t r = 0; r < batch_size; ++r)
                batch[i][r] = whole[i][n + r];

        result.emplace_back(batch);
    }

    if (whole.get_len2() % batch_size == 0)
        return result;

    Tensor2 rem(whole.get_len1(),
                whole.get_len2() % batch_size);

    for (std::size_t i = 0; i < whole.get_len1(); ++i)
        for (std::size_t r = quo; r < whole.get_len2(); ++r)
            rem[i][r - quo] = whole[i][r];

    result.emplace_back(rem);
    return result;
}

void write_label(const Tensor2 &label,
                 const std::string &filename)
{
    std::ios_base::sync_with_stdio(false);
    std::ofstream fout(filename.c_str());

    for (std::size_t r = 0; r < label.get_len2(); ++r)
    {
        float val = 0;
        std::size_t max = 0;

        for (std::size_t i = 0; i < label.get_len1(); ++i)
            if (label[i][r] > val)
                val = label[i][r], max = i;

        fout << max << "\n";
    }

    fout.close();
}


int main(void)
{
    std::size_t batch_size = 100;
    std::vector<Tensor2> biases, weights;
    Tensor2 X_test = read_csv("X_test.csv",
                              false, 28 * 28, 10000),
            y_test = read_csv("y_test.csv",
                              true, 10, 10000);
    X_test /= 255;

    {
        Layers layers;
        layers.add_layer(28 * 28, nullptr, nullptr);
        layers.add_layer(300, std::tanh, deriv_tanh);
        layers.add_layer(100, std::tanh, deriv_tanh);
        layers.add_layer(10, nullptr, nullptr);
        layers.initialize(batch_size);
        layers.initialize_deltas();
        std::vector<Tensor2> X_batches, y_batches;

        {
            Tensor2 X_train = read_csv("X_train_full.csv",
                                       false, 28 * 28, 60000),
                    y_train = read_csv("y_train_full.csv",
                                       true, 10, 60000);
            X_train /= 255;
            X_batches = divide(X_train, batch_size);
            y_batches = divide(y_train, batch_size);
        }

        std::cout << "Reading from file completed.\n"
                  << std::endl;
        float learning_rate = 1.0;

        for (std::size_t epoch = 0; epoch < 20; ++epoch)
        {
            std::size_t idx;

            for (idx = 0; idx < X_batches.size() - 1; ++idx)
            {
                layers.feed_for(X_batches[idx]);
                layers.prop_back(X_batches[idx], y_batches[idx]);
                layers.desc_grad(learning_rate, batch_size);
                std::cout << "  Epoch " << epoch + 1
                          << ", iter " << idx + 1 << " done."
                          << std::endl;
            }

            layers.feed_for(X_batches[idx]);
            layers.prop_back(X_batches[idx], y_batches[idx]);
            layers.desc_grad(learning_rate,
                             X_batches[idx].get_len2());
            std::cout << "  Epoch " << epoch + 1 << " done.\n"
                      << std::endl;

            {
                Layers tester;
                tester.add_layer(28 * 28, nullptr, nullptr);
                tester.add_layer(300, std::tanh, deriv_tanh);
                tester.add_layer(100, std::tanh, deriv_tanh);
                tester.add_layer(10, nullptr, nullptr);
                tester.initialize(10000,
                                  layers.get_biases(),
                                  layers.get_weights());
                std::stringstream ss;
                ss << "y_pred_" << epoch + 1 << ".csv";
                tester.feed_for(X_test);
                write_label(tester.get_prediction(), ss.str());
            }
        }
    }

    std::system("pause");
    return 0;
}