import numpy as np
import matplotlib.pyplot as plt
import sinClass


def h(t,T,fc) :

    # 교제 p523 식7.10 
    return 2*fc*T*np.sinc(2*fc*t)   # sinc(x) = sin(pi*x)/(pi*x)

def interpolation_func(t,x,max_n,T,fc) :
    # 교제 p523 식7.9
    result = 0
    for n in range(0,max_n) :
        result += (x[n]*h(t-n*T, T, fc)) # x(n*T)는 x[n]과 같음

    return result
#--------------- 아래부터 시작 --------------------


#-------------- original signal 생성 --------
startTime = 0
endTime = 1
sampleGap = 0.001
t = np.arange(startTime,endTime,sampleGap) # 0부터 1까지 0.001간격으로 벡터 생성

resultTest1 = np.sin(2*np.pi*t) # sin(2*pi*t)
resultTest2 = 2*np.sin(2*np.pi*5*t) # 2*sin(2*pi*5*t)
resultTest3 = 4*np.sin(2*np.pi*10*t) # 4*sin(2*pi*10*t)
y = resultTest1 + resultTest2 + resultTest3 # sin(2*pi*t) + 2*sin(2*pi*5*t) + 4*sin(2*pi*10*t)


#------------ original signal Ts간격으로 샘플링 ----
Ts = 0.01 #sampling time 
n_max = 100
t_sampling = np.arange(startTime,endTime,Ts)
y_sampling = np.zeros(n_max)
for n in range(n_max) :
    y_sampling[n] = y[n*10]  

#---------------- 보간법 ------------------------
fc = 50 #cut-off frequency
r = interpolation_func(t,y_sampling,n_max,Ts,fc)



#------------- original signal 그래프 그리기
fig, ax = plt.subplots(3,1)
ax[0].plot(t,y,'r')
ax[0].set_title('original signal')
ax[0].set_xlabel('time')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)

#------------ 샘플링된 신호 그래프 그리기
ax[1].plot(t_sampling,y_sampling,'g',linestyle = ' ', marker='.')
ax[1].set_title('sampled sigmal')
ax[1].set_xlabel('time')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)

#----------- 보간법으로 복원한 신호 그리기
ax[2].plot(t,r,'b')
ax[2].set_title('interpolation result')
ax[2].set_xlabel('time')
ax[2].set_ylabel('Amplitude')
ax[2].grid(True)


plt.show() #위의 값으로 그린 그래프 화면에 띄우기


# 질문은 19511호로 와주세요.



