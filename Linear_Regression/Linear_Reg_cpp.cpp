#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
using namespace std;

void grad_descent(vector<int> x, vector<int> y)
{
    double m_cur = 0, b_cur = 0;
    int iterations = 100;
    double learning_rate = 0.003;
    double n = x.size();
    for(int j=0; j<iterations; j++)
    {
        vector<double> y_pred;
        for(auto i: x)
            y_pred.push_back(m_cur * i + b_cur);
        vector<double> y_diff;
        for(int i = 0; i < y.size(); i++)
            y_diff.push_back(y[i] - y_pred[i]);
        
        vector<double> mult_x;
        for(int i = 0; i < x.size(); i++)
            mult_x.push_back(x[i] * y_diff[i]);
        double cost = (1/n) * pow(accumulate(y_diff.begin(), y_diff.end(), 0), 2);
        double md = -(2/n) * accumulate(mult_x.begin(), mult_x.end(), 0);
        double bd = -(2/n) * accumulate(y_diff.begin(), y_diff.end(), 0);

        m_cur = m_cur - learning_rate * md;
        b_cur = b_cur - learning_rate * bd;
        cout << "m:" << m_cur  << "  " << "b:" << b_cur << "  " << "cost:" << cost << "  " << "iter:" << j << endl;
    } 
}

int main() 
{
    vector<int> x{ 1, 2, 3, 4, 5 };
    vector<int> y{ 5, 7, 9, 11, 13 };
    grad_descent(x, y);
}