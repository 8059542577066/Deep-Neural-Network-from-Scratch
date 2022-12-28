#include "model.h"
#include <cmath>
#include <random>


float sigmoid(float x)
{
    return 1 / (1 + std::exp(-x));
}

float deriv_sigmoid(float y)
{
    return y * (1 - y);
}

float deriv_tanh(float y)
{
    return (1 + y) * (1 - y);
}


Tensor2::Tensor2()
    : len1(0), len2(0), capa(0),
      array(nullptr), temp_soft(nullptr) {}

Tensor2::Tensor2(std::size_t len1, std::size_t len2)
    : len1(len1), len2(len2), capa(len1 * len2),
      array(new float[len1 * len2]), temp_soft(nullptr) {}

Tensor2::Tensor2(const Tensor2 &other)
    : len1(other.len1), len2(other.len2), capa(other.len1 * other.len2),
      array(new float[other.len1 * other.len2]), temp_soft(nullptr)
{
    for (std::size_t idx = 0; idx < this->len1 * this->len2; ++idx)
        this->array[idx] = other.array[idx];
}

Tensor2::~Tensor2()
{
    delete[] this->array;
    delete[] this->temp_soft;
}

float *Tensor2::operator[](std::size_t idx1)
{
    return &this->array[idx1 * this->len2];
}

float *Tensor2::operator[](std::size_t idx1) const
{
    return &this->array[idx1 * this->len2];
}

Tensor2 &Tensor2::operator-=(const Tensor2 &other)
{
    if (this->len1 != other.len1 || this->len2 != other.len2)
        throw DIM_MISMATCH;

    for (std::size_t idx = 0; idx < this->len1 * this->len2; ++idx)
        this->array[idx] -= other.array[idx];

    return *this;
}

Tensor2 &Tensor2::operator*=(float value)
{
    for (std::size_t idx = 0; idx < this->len1 * this->len2; ++idx)
        this->array[idx] *= value;

    return *this;
}

Tensor2 &Tensor2::operator/=(float value)
{
    for (std::size_t idx = 0; idx < this->len1 * this->len2; ++idx)
        this->array[idx] /= value;

    return *this;
}

void Tensor2::resize(std::size_t len1, std::size_t len2)
{
    if (len1 * len2 > this->capa)
    {
        delete[] this->array;
        this->array = new float[len1 * len2];
        this->capa = len1 * len2;
    }

    this->len1 = len1, this->len2 = len2;
}

void Tensor2::resize_soft()
{
    delete[] this->temp_soft;
    this->temp_soft = new float[this->len1 * this->len2];
}

void Tensor2::apply(float (*func)(float))
{
    for (std::size_t idx = 0; idx < this->len1 * this->len2; ++idx)
        this->array[idx] = (*func)(this->array[idx]);
}

void Tensor2::prep_softmax(std::size_t r)
{
    this->temp_soft[r] = 0;

    for (std::size_t s = 0; s < this->len1; ++s)
        this->temp_soft[r] += (*this)[s][r];
}

float Tensor2::softmax(std::size_t j, std::size_t r) const
{
    return (*this)[j][r] / this->temp_soft[r];
}

std::size_t Tensor2::get_len1() const
{
    return this->len1;
}

std::size_t Tensor2::get_len2() const
{
    return this->len2;
}


void Layers::add_layer(std::size_t size,
                       float (*func)(float),
                       float (*deriv)(float))
{
    this->sizes.emplace_back(size);
    this->funcs.emplace_back(func);
    this->derivs.emplace_back(deriv);
}

void Layers::initialize(std::size_t batch_size)
{
    this->biases.emplace_back();

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
        this->biases.emplace_back(this->sizes[l], 1);

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
        for (std::size_t j = 0; j < this->sizes[l]; ++j)
            this->biases[l][j][0] = 0;

    this->weights.emplace_back();

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
        this->weights.emplace_back(this->sizes[l - 1],
                                   this->sizes[l]);

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
    {
        float sig = std::sqrt(2.0 / (this->sizes[l - 1] +
                                     this->sizes[l]));
        std::default_random_engine generator;
        std::normal_distribution<float> dist(0, sig);

        for (std::size_t i = 0; i < this->sizes[l - 1]; ++i)
            for (std::size_t j = 0; j < this->sizes[l]; ++j)
                this->weights[l][i][j] = dist(generator);
    }

    this->activs.emplace_back();

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
        this->activs.emplace_back(this->sizes[l], batch_size);

    this->activs.rbegin()->resize_soft();
}

void Layers::initialize_deltas()
{
    this->delta_biases.emplace_back();

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
        this->delta_biases.emplace_back(this->sizes[l], 1);

    this->delta_weights.emplace_back();

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
        this->delta_weights.emplace_back(this->sizes[l - 1],
                                         this->sizes[l]);
}

void Layers::initialize(std::size_t batch_size,
                        const std::vector<Tensor2> &biases,
                        const std::vector<Tensor2> &weights)
{
    this->biases = biases, this->weights = weights;
    this->activs.emplace_back();

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
        this->activs.emplace_back(this->sizes[l], batch_size);

    this->activs.rbegin()->resize_soft();
}

void Layers::feed_for(const Tensor2 &input)
{
    if (this->sizes.size() < 2)
        throw NO_OUTPUT_LAYER;

    std::size_t batch_size = input.get_len2();

    for (std::size_t j = 0; j < this->sizes[1]; ++j)
        for (std::size_t r = 0; r < batch_size; ++r)
        {
            this->activs[1][j][r] = this->biases[1][j][0];

            for (std::size_t i = 0; i < this->sizes[0]; ++i)
                this->activs[1][j][r] += this->weights[1][i][j] *
                                         input[i][r];
        }

    if (this->sizes.size() > 2)
        this->activs[1].apply(this->funcs[1]);

    for (std::size_t l = 2; l < this->sizes.size() - 1; ++l)
    {
        for (std::size_t j = 0; j < this->sizes[l]; ++j)
            for (std::size_t r = 0; r < batch_size; ++r)
            {
                this->activs[l][j][r] = this->biases[l][j][0];

                for (std::size_t i = 0; i < this->sizes[l - 1]; ++i)
                    this->activs[l][j][r] += this->weights[l][i][j] *
                                             this->activs[l - 1][i][r];
            }

        this->activs[l].apply(this->funcs[l]);
    }

    std::size_t L = this->sizes.size() - 1;

    if (this->sizes.size() > 2)
        for (std::size_t j = 0; j < this->sizes[L]; ++j)
            for (std::size_t r = 0; r < batch_size; ++r)
            {
                this->activs[L][j][r] = this->biases[L][j][0];

                for (std::size_t i = 0; i < this->sizes[L - 1]; ++i)
                    this->activs[L][j][r] += this->weights[L][i][j] *
                                             this->activs[L - 1][i][r];
            }

    this->activs[L].apply(std::exp);

    for (std::size_t r = 0; r < batch_size; ++r)
    {
        this->activs[L].prep_softmax(r);

        for (std::size_t j = 0; j < this->sizes[L]; ++j)
            this->activs[L][j][r] = this->activs[L].softmax(j, r);
    }
}

void Layers::prop_back(const Tensor2 &input, const Tensor2 &label)
{
    if (this->sizes.size() < 2)
        throw NO_OUTPUT_LAYER;

    if (input.get_len2() != label.get_len2())
        throw BATCH_MISMATCH;

    std::size_t batch_size = input.get_len2(),
                l = this->sizes.size() - 1;
    this->errors.resize(this->sizes[l], batch_size);

    for (std::size_t j = 0; j < this->sizes[l]; ++j)
        for (std::size_t r = 0; r < batch_size; ++r)
            this->errors[j][r] = (this->activs[l][j][r] - label[j][r]) *
                                 deriv_sigmoid(this->activs[l][j][r]);

    for (std::size_t j = 0; j < this->sizes[l]; ++j)
    {
        this->delta_biases[l][j][0] = 0;

        for (std::size_t r = 0; r < batch_size; ++r)
            this->delta_biases[l][j][0] += this->errors[j][r];
    }

    if (this->sizes.size() > 2)
    {
        for (std::size_t i = 0; i < this->sizes[l - 1]; ++i)
            for (std::size_t j = 0; j < this->sizes[l]; ++j)
            {
                this->delta_weights[l][i][j] = 0;

                for (std::size_t r = 0; r < batch_size; ++r)
                    this->delta_weights[l][i][j] += this->errors[j][r] *
                                                    this->activs[l - 1][i][r];
            }

        this->delta_activs.resize(this->sizes[l - 1], batch_size);

        for (std::size_t i = 0; i < this->sizes[l - 1]; ++i)
            for (std::size_t r = 0; r < batch_size; ++r)
            {
                this->delta_activs[i][r] = 0;

                for (std::size_t j = 0; j < this->sizes[l]; ++j)
                    this->delta_activs[i][r] += this->errors[j][r] *
                                                this->weights[l][i][j];
            }
    }

    for (--l; l > 1; --l)
    {
        this->errors.resize(this->sizes[l], batch_size);

        for (std::size_t j = 0; j < this->sizes[l]; ++j)
            for (std::size_t r = 0; r < batch_size; ++r)
                this->errors[j][r] = this->delta_activs[j][r] *
                                     this->derivs[l](this->activs[l][j][r]);

        for (std::size_t j = 0; j < this->sizes[l]; ++j)
        {
            this->delta_biases[l][j][0] = 0;

            for (std::size_t r = 0; r < batch_size; ++r)
                this->delta_biases[l][j][0] += this->errors[j][r];
        }

        for (std::size_t i = 0; i < this->sizes[l - 1]; ++i)
            for (std::size_t j = 0; j < this->sizes[l]; ++j)
            {
                this->delta_weights[l][i][j] = 0;

                for (std::size_t r = 0; r < batch_size; ++r)
                    this->delta_weights[l][i][j] += this->errors[j][r] *
                                                    this->activs[l - 1][i][r];
            }

        this->delta_activs.resize(this->sizes[l - 1], batch_size);

        for (std::size_t i = 0; i < this->sizes[l - 1]; ++i)
            for (std::size_t r = 0; r < batch_size; ++r)
            {
                this->delta_activs[i][r] = 0;

                for (std::size_t j = 0; j < this->sizes[l]; ++j)
                    this->delta_activs[i][r] += this->errors[j][r] *
                                                this->weights[l][i][j];
            }
    }

    if (this->sizes.size() > 2)
    {
        this->errors.resize(this->sizes[1], batch_size);

        for (std::size_t j = 0; j < this->sizes[1]; ++j)
            for (std::size_t r = 0; r < batch_size; ++r)
                this->errors[j][r] = this->delta_activs[j][r] *
                                     this->derivs[1](this->activs[1][j][r]);

        for (std::size_t j = 0; j < this->sizes[1]; ++j)
        {
            this->delta_biases[1][j][0] = 0;

            for (std::size_t r = 0; r < batch_size; ++r)
                this->delta_biases[1][j][0] += this->errors[j][r];
        }
    }

    for (std::size_t i = 0; i < this->sizes[0]; ++i)
        for (std::size_t j = 0; j < this->sizes[1]; ++j)
        {
            this->delta_weights[1][i][j] = 0;

            for (std::size_t r = 0; r < batch_size; ++r)
                this->delta_weights[1][i][j] += this->errors[j][r] *
                                                input[i][r];
        }
}

void Layers::desc_grad(float learning_rate, std::size_t divisor)
{
    for (std::size_t l = 1; l < this->sizes.size(); ++l)
    {
        this->delta_biases[l] *= learning_rate / divisor;
        this->biases[l] -= this->delta_biases[l];
    }

    for (std::size_t l = 1; l < this->sizes.size(); ++l)
    {
        this->delta_weights[l] *= learning_rate / divisor;
        this->weights[l] -= this->delta_weights[l];
    }
}

std::vector<Tensor2> Layers::get_biases() const
{
    return this->biases;
}

std::vector<Tensor2> Layers::get_weights() const
{
    return this->weights;
}

Tensor2 Layers::get_prediction() const
{
    return *this->activs.crbegin();
}