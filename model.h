#ifndef MODEL_H
#define MODEL_H


#ifdef __WIN32__
#ifdef BUILD_LIB
#define LIB_CLASS __declspec(dllexport)
#else
#define LIB_CLASS __declspec(dllimport)
#endif
#else
#define LIB_CLASS
#endif


#ifndef DIM_MISMATCH
#define DIM_MISMATCH 1
#endif

#ifndef NO_OUTPUT_LAYER
#define NO_OUTPUT_LAYER 11
#endif

#ifndef BATCH_MISMATCH
#define BATCH_MISMATCH 12
#endif


#include <vector>

float LIB_CLASS sigmoid(float);
float LIB_CLASS deriv_sigmoid(float);
float LIB_CLASS deriv_tanh(float);


class LIB_CLASS Tensor2
{
    std::size_t len1, len2, capa;
    float *array;
    float *temp_soft;

public:
    Tensor2();
    Tensor2(std::size_t, std::size_t);
    Tensor2(const Tensor2 &);
    ~Tensor2();

    float *operator[](std::size_t);
    float *operator[](std::size_t) const;

    Tensor2 &operator-=(const Tensor2 &);
    Tensor2 &operator*=(float);
    Tensor2 &operator/=(float);

    void resize(std::size_t, std::size_t);
    void resize_soft();
    void apply(float (*)(float));
    void prep_softmax(std::size_t);

    float softmax(std::size_t, std::size_t) const;

    std::size_t get_len1() const;
    std::size_t get_len2() const;
};


class LIB_CLASS Layers
{
    std::vector<std::size_t> sizes;

    std::vector<Tensor2> biases;
    std::vector<Tensor2> weights;
    std::vector<Tensor2> activs;
    std::vector<float (*)(float)> funcs;
    std::vector<float (*)(float)> derivs;

    std::vector<Tensor2> delta_biases;
    std::vector<Tensor2> delta_weights;
    Tensor2 delta_activs;
    Tensor2 errors;

public:
    void add_layer(std::size_t,
                   float (*)(float),
                   float (*)(float));
    void initialize(std::size_t);
    void initialize_deltas();
    void initialize(std::size_t,
                    const std::vector<Tensor2> &,
                    const std::vector<Tensor2> &);

    void feed_for(const Tensor2 &);
    void prop_back(const Tensor2 &, const Tensor2 &);
    void desc_grad(float, std::size_t);

    std::vector<Tensor2> get_biases() const;
    std::vector<Tensor2> get_weights() const;

    Tensor2 get_prediction() const;
};


#endif