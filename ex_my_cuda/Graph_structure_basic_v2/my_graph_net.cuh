#ifndef __MY_GRAPH_NET_CU__
#define __MY_GRAPH_NET_CU__

#include "my_graph_net_sub.cuh"

//---------------------------------------
class MY_GRAPH_NET{

    private :
        MY_GRAPH_NET_DEQUE deque_operate; 
        MY_MATRIX_DEQUE deque_matrix;

    protected :


    public :
        cublasHandle_t handle;
        curandGenerator_t rand_gen;
        MY_MATRIX_DEVICE d_ones;
        MY_MATRIX_DEVICE d_temp;


        MY_GRAPH_NET();
        ~MY_GRAPH_NET();


        void network_init(int seed = 0);
        
        void foreward();
        void test();
        void backward();
        //---------------------------------------

        MY_MATRIX_DEVICE* multi(MY_MATRIX_DEVICE *pmatrix1, MY_MATRIX_DEVICE *pmatrix2);
        MY_MATRIX_DEVICE* add_bias(MY_MATRIX_DEVICE *pmatrix1, MY_MATRIX_DEVICE *pbias);
 
        MY_MATRIX_DEVICE* adding_point(MY_MATRIX_DEVICE *pmatrix1, MY_MATRIX_DEVICE *pmatrix2);
        void dividing_point(MY_MATRIX_DEVICE *str, MY_MATRIX_DEVICE **dst1, MY_MATRIX_DEVICE **dst2);
        
 
        MY_MATRIX_DEVICE* merge(MY_MATRIX_DEVICE *pmatrix1, MY_MATRIX_DEVICE *pmatrix2); //[matrix1 ; matrix2] concatenate row
        MY_MATRIX_DEVICE* stack(MY_MATRIX_DEVICE *pmatrix1, MY_MATRIX_DEVICE *pmatrix2); //[matrix1 , matrix2] concatenate column
        
        MY_MATRIX_DEVICE* rand_scale(MY_MATRIX_DEVICE *pmatrix1, float from = 0.0, float to = 1.0);
        MY_MATRIX_DEVICE* scale(MY_MATRIX_DEVICE *pmatrix1, float scala);
       

        
        
        MY_MATRIX_DEVICE* inverted_dropout(MY_MATRIX_DEVICE *pmatrix1, float rate);
         
 
        MY_MATRIX_DEVICE* uniform_noise(int row, int column); 
        MY_MATRIX_DEVICE* white_noise(int row, int column,float rate = 0.5);      
        MY_MATRIX_DEVICE* min(MY_MATRIX_DEVICE *pmatrix, float max_value);
         


        MY_MATRIX_DEVICE* elu(MY_MATRIX_DEVICE *);
        MY_MATRIX_DEVICE* relu(MY_MATRIX_DEVICE *);
        MY_MATRIX_DEVICE* sigmoid(MY_MATRIX_DEVICE *);
        MY_MATRIX_DEVICE* tanh(MY_MATRIX_DEVICE *);

        float sum_absolute(MY_MATRIX_DEVICE *);
        float average_absolute(MY_MATRIX_DEVICE *);
        float accuracy(MY_MATRIX_DEVICE *y, MY_MATRIX_DEVICE *t);

        MY_MATRIX_DEVICE* binary_cross_entropy(MY_MATRIX_DEVICE *presult, MY_MATRIX_DEVICE *ptarget);        
        MY_MATRIX_DEVICE* least_squares(MY_MATRIX_DEVICE *presult, MY_MATRIX_DEVICE *ptarget);





};


class MY_MOMENTUM_OPTIMIZER{
    private :
        MY_GRAPH_NET_DEQUE deque_operate; 
        MY_MATRIX_DEQUE deque_matrix;
    public :
        float learning_rate;
        float momentum_rate;

        MY_MOMENTUM_OPTIMIZER();
        ~MY_MOMENTUM_OPTIMIZER();
        void set_hyperpara(float l_rate = 0.1,float m_rate = 0.8);
        void set_para(MY_MATRIX_DEVICE *pa, ...);
        void update();

};


class MY_ADAM_OPTIMIZER{
    private :
        MY_GRAPH_NET_DEQUE deque_operate; 
        MY_MATRIX_DEQUE deque_matrix;
    public :
        float learning_rate;
        float beta1;
        float beta2;
        float beta1_t;
        float beta2_t;

        MY_ADAM_OPTIMIZER();
        ~MY_ADAM_OPTIMIZER();
        void set_hyperpara(float l_rate = 0.0001,float beta1_rate = 0.9, float beta2_rate = 0.999);
        void set_para(MY_MATRIX_DEVICE *pa, ...);
        void update();

};


enum REGULARIZATION_STAT{
    L1_NORM = 0,
    L2_NORM,
    MAX_NORM

};


class MY_REGULARIZATION{
    private :
        MY_GRAPH_NET_DEQUE deque_operate; 
    public :
        float L1_rate;
        float L2_rate;
        float max_rate;

        MY_REGULARIZATION();
        ~MY_REGULARIZATION();
        void set_para(REGULARIZATION_STAT stat, float rate,  ...);
        void update();


};


#endif
