#ifndef __MY_DEVICE_FUNC_CU__
#define __MY_DEVICE_FUNC_CU__


#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> column : x, i -> row : y, column major


__global__ void make_ones(float *a, int n);
__global__ void make_zeros(float *a, int n);
__global__ void transpose(float *dst, const float *str,int str_row, int str_column, int n);
__global__ void add_bias(const float *a,const float *b, float *c, int a_row, int n);
__global__ void transfer(float *dst,const float *str,int n);
__global__ void relu(float *dst,const float *str,int n);
__global__ void relu_inv(float *dst,const float *str,int n);
__global__ void scalar_multi(const float *a, const float *b, float *c, int n);
__global__ void elu(float *dst,const float *str,int n);
__global__ void elu_inv(float *dst,const float *str,int n);
__global__ void binary_cross_entropy(const float *a,const float *b,float *c, int n);
__global__ void binary_cross_entropy_inv(const float *a,const float *b,float *c, int n);
__global__ void dropout_table(float *a, float dropout_rate,int n);
__global__ void dropout(const float *a,const float *b, float *c, float dropout_rate,int n);
__global__ void sigmoid(float *dst,const float *str,int n);
__global__ void sigmoid_inv(float *dst,const float *str,int n);
__global__ void tanh(float *dst,const float *str,int n);
__global__ void tanh_inv(float *dst,const float *str,int n);
__global__ void add(const float *a, const float *b, float *c, int n);
__global__ void least_squares(const float *a,const float *b,float *c, int n);
__global__ void least_squares_inv(const float *a,const float *b,float *c, int n);
__global__ void momentum_vector(const float *a, float *b, float l_rate,float m_rate, int n);
__global__ void adam_beta1(const float *a, float *b, float beta1, int n);
__global__ void adam_beta2(const float *a, float *b, float beta2, int n);
__global__ void adam_sum(const float *a, const float *b, float *c, float learning_rate,float beta1_t,float beta2_t, int n);
__global__ void max_norm(float *a, float rate, int n);
__global__ void min(float *dst,const float *str, float rate, int n);
__global__ void min_inv(float *dst,const float *str,float rate, int n);
__global__ void accuracy_table(const float *y, const float *t,float *r, int row ,int column);


#endif

