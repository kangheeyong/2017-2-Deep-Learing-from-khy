#ifndef __MY_DEVICE_FUNCTION_CU__
#define __MY_DEVICE_FUNCTION_CU__

#define IDX2C(i,j,Id)       (((j)*(Id))+(i)) // j -> row, i -> column



__global__ void sigmoid(float *a,float *z,long n);
__global__ void sigmoid_inv(float *a,float *z,long n);

__global__ void relu(float *a,float *z,long n);
__global__ void relu_inv(float *a,float *z,long n);

__global__ void new_activation(float *a,float *z,long n);
__global__ void new_activation_inv(float *a,float *z,long n);




// front -> rear 변수에 전달 
__global__ void deliver_front_to_rear(float *front,float *rear,long n);
// z = z + b
__global__ void add_bias(float *z,float *b,long column,long n);
// temp = (y-T)*(2*batch_size) 
__global__ void last_delta_before_transpose(float *temp, float *y,float *T,long batch_size,long n);
// before -> after transpose
__global__ void transpose(float *after, float *before,long before_columns,long before_rows);
// c = a .* b
__global__ void basic_multi(float *a,float *b,float *c, long n);
//temp = -0.5*(T*log(y) + (1-T)*log(1-y))
__global__ void loss_cross_entropy(float *target,float *y, float * result,long last_neural,long batch_size);
// result = matching(target, y) -> 같으면 1 아니면 0
__global__ void matching(float *target,float *y, float * result,long last_neural,long batch_size);
// w = w - alpha*(delta_w + ramda*w)
__global__ void weight_update(float *w,float *delta_w, float alpha,float ramda,long n);
// b = b - alpha * delta_b
__global__ void bias_update(float *b,float *delta_b, float alpha,long n);
// init 0
__global__ void init_zeros(float *a, long n);
// init 1
__global__ void init_ones(float *a, long n);
//adam_mean(i) = beta1*adam_mean(i) + (1-beta1)*delta(i)  
__global__ void adam_mean(float *adam_mean, float* delta, float beta1,long n);
//adam_var(i) = beta2*adam_var(i) + (1-beta2)*deilta(i)  
__global__ void adam_var(float *adam_var, float* delta, float beta2,long n);
 //temp = adam_mean(i)/(1-beta1_t)/(sqrt(adam_var/(1-beta2_t)) + 0.00000001)
__global__ void adam_sum(float *result, float *adam_mean,float *adam_var,float beta1_t,float beta2_t,long n);
__global__ void maxnorm_constraints(float *a, float max,long n);

__global__ void inverted_dropout(float *dropout,float *probability, float dropout_rate,long n);


#endif
