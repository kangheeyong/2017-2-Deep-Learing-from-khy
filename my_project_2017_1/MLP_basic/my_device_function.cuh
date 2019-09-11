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


#endif
