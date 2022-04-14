#include <string.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <tuple>
#include <fstream>
#include <typeinfo>
#include <math.h>
#include <random>

using namespace std;
typedef double T;


// Why Segmentation Error ??? Make sure input file location is correct ...



bool contains(char arr[], string s){
    char c = s[0];
    bool contain = false;
    for (int i=0; i<strlen(arr);i++){
        if (c==arr[i]){
            contain = true;
            break;
        }
        }
    return contain;
}

char* ops(string i, char* variables, char* operations_1, char* operations_2){
    if (i=="0") {return variables;}
    if (i=="1") {return operations_1;}
    if (i=="2") {return operations_2;}
};

vector<int> each_ids(int flat_id, vector<int> prod_dims){
    int len = prod_dims.size();
    vector<int> each_id;
    for (int i=0;i<len;i++){
        int de = prod_dims[len-i-1];
        int q = flat_id/de;
        flat_id = flat_id - q*de;
        each_id.push_back(q);
    }
    return each_id;
}


T H_evaluate(vector<char> RPN, int arity_len, vector<T> z){
    vector<T> stacks;
    int j = 0;
    for (int k=0; k<arity_len; k++){
    char sym = RPN[k];
    if (sym=='+') {
        stacks[j-2] = stacks[j-2]+stacks[j-1];
        j = j-1;
        stacks.pop_back();
        }
    if (sym=='-') {
        stacks[j-2] = stacks[j-2]-stacks[j-1];
        j = j-1;
        stacks.pop_back();
        }
    if (sym=='*') {
        stacks[j-2] = stacks[j-2]*stacks[j-1];
        j = j-1;
        stacks.pop_back();
        }
    if (sym=='/') {
        stacks[j-2] = stacks[j-2]/stacks[j-1];
        j = j-1;
        stacks.pop_back();
        }
    if (sym=='>') {
         stacks[j-1] = stacks[j-1] + 1;
        }
    if (sym=='<') {
         stacks[j-1] = stacks[j-1] - 1;
        }
    if (sym=='~') {
         stacks[j-1] = - stacks[j-1];
        }
    if (sym=='O') {
         stacks[j-1] = 2*stacks[j-1];
        }
    if (sym=='Q') {
         stacks[j-1] = pow(stacks[j-1],2);
        }
    if (sym=='L') {
         stacks[j-1] = log(stacks[j-1]);
        }
    if (sym=='E') {
         stacks[j-1] = exp(stacks[j-1]);
        }
    if (sym=='S') {
         stacks[j-1] = sin(stacks[j-1]);
        }
    if (sym=='C') {
         stacks[j-1] = cos(stacks[j-1]);
        }
    if (sym=='A') {
         stacks[j-1] = abs(stacks[j-1]);
        }
    if (sym=='N') {
         stacks[j-1] = asin(stacks[j-1]);
        }
    if (sym=='T') {
         stacks[j-1] = atan(stacks[j-1]);
        }
    if (sym=='R') {
         stacks[j-1] = sqrt(stacks[j-1]);
        }
    if (sym=='I') {
         stacks[j-1] = 1/stacks[j-1];
        }
    if (sym=='P') {
         stacks.push_back(atan(1)*4);
         j = j+1;
        }
    if (sym=='a') {
         stacks.push_back(z[0]);
         j = j+1;
        }
    if (sym=='b') {
         stacks.push_back(z[1]);
         j = j+1;
        }
    if (sym=='c') {
         stacks.push_back(z[2]);
         j = j+1;
        }
    if (sym=='d') {
         stacks.push_back(z[3]);
         j = j+1;
        }
    if (sym=='e') {
         stacks.push_back(z[4]);
         j = j+1;
        }
    if (sym=='f') {
         stacks.push_back(z[5]);
         j = j+1;
        }
    }
    return stacks[0];
}



T norm(vector<T> vec, int dim){
    T squared = 0.0;
    for (int i=0;i<dim;i++){
        squared = squared + pow(vec[i],2);
    }
    return sqrt(squared);

}vector<T> normalized_vec(vector<T> vec, int dim){
    T norm_vec = norm(vec, dim);
    vector<T> vecp = vec;
    if (norm_vec<1e-12) {
    for (int i=0;i<dim;i++){
        vecp[i] = 1.0;
        /*throw 20;*/
    }
    norm_vec = norm(vecp, dim);
    }
    for (int i=0;i<dim;i++){
        vecp[i] = vecp[i]/norm_vec;
    }
    return vecp;
}





vector<T> random_vec(const int dim){
    std::random_device rd{};
    std::mt19937 gen{rd()};
    /*std::default_random_engine generator;*/
    std::normal_distribution<double> distribution(0.0,1.0);
    vector<T> rv;
    for (int i=0;i<dim;i++){
        T number = distribution(gen);
        rv.push_back(number);
    }
    rv = normalized_vec(rv, dim);
    return rv;
}


vector<T> H_grad_evaluate(vector<char> RPN, int arity_len, vector<T> z, const int dim){
    T eps = 1e-6;
    vector<T> grads;
    for (int i=0;i<dim;i++){
        vector<T> zp = z;
        vector<T> zm = z;
        zp[i] = zp[i] + eps;
        zm[i] = zm[i] - eps;
        T Hp = H_evaluate(RPN, arity_len, zp);
        T Hm = H_evaluate(RPN, arity_len, zm);
        T grad = (Hp-Hm)/(2*eps);
        grads.push_back(grad);
    }
    return grads;
}

T my_inner_product(vector<T> vec1,vector<T> vec2, int dim){
    T eps = 0.0;
    for (int i=0;i<dim;i++){
        eps = eps + vec1[i]*vec2[i];
    }
    return eps;
}

vector<T> svprod(T s, vector<T> vec){
    int dim = vec.size();
    vector<T> vec2;
    for (int i=0;i<dim;i++){
        vec2.push_back(s*vec[i]);
    }
    return vec2;
}

vector<T> vsub(vector<T> v1, vector<T> v2){
    int dim = v1.size();
    vector<T> v3;
    for (int i=0;i<dim;i++){
        v3.push_back(v1[i]-v2[i]);
    }
    return v3;
}

void printvec(vector<T> vec){
    int num = vec.size();
    for (int i=0;i<num;i++){
        cout << vec[i] << endl;
    }
}

void printrpn(vector<char> RPN){
    int num = RPN.size();
    for (int i=0;i<num;i++){
        cout << RPN[i];
    }
    cout << endl;
}



vector<T> schmidt(vector<T> vec, const int dim){
    /* vec does not have to be normalized*/
    int num_vecs = vec.size()/dim;
    for (int i=0;i<num_vecs;i++){
        /*vector, how to get a slice*/
        vector<T> v = std::vector<T>(vec.begin()+i*dim, vec.begin()+(i+1)*dim);
        for (int j=0;j<i;j++){
            vector<T> v2 = std::vector<T>(vec.begin()+j*dim, vec.begin()+(j+1)*dim);
            v = vsub(v, svprod(my_inner_product(v,v2,dim),v2));
        }
        v = normalized_vec(v, dim);
        for (int k=0;k<dim;k++){
            vec[i*dim+k] = v[k];
        }
    }
    return vec;
    /*If no return, segmentation error*/
}

vector<T> project_orth(vector<T> vec, vector<T> bases, const int dim){
    /* bases should be normalized and orthogonal. If not, use schmidt first */
    int num_bases = bases.size()/dim;
    for (int i=0;i<num_bases;i++){
        vector<T> v = std::vector<T>(bases.begin()+i*dim, bases.begin()+(i+1)*dim);
        vec = vsub(vec, svprod(my_inner_product(vec,v,dim),v));
    }
    vec = normalized_vec(vec, dim);
    return vec;
}


int test3(){
    vector<T> vec;
    vec.push_back(1.00);
    vec.push_back(0.00);
    vec.push_back(0.00);
    vec.push_back(2.00);
    vec.push_back(1.00);
    vec.push_back(0.00);
    vector<T> test_vec;
    test_vec.push_back(2.00);
    test_vec.push_back(1.00);
    test_vec.push_back(1.00);
    printvec(project_orth(test_vec, schmidt(vec, 3), 3));
}


int test2(){
    vector <T> rv1 = random_vec(2);
    vector <T> rv2 = random_vec(2);
    cout << rv1[0];
    cout << rv1[1];
    cout << rv2[0];
    cout << rv2[1];
}

int main(){
    /*string env = "harmonic_1d";*/
    /*string env = "harmonic_2d_iso";*/
    /*string env = "harmonic_2d_aniso";*/
    string env = "kepler_2d";
    /*string env = "haha";*/
    const int input_dim = 5;
    int test_pts = 3;
    char variables2[] = {'a','b','c','d','e','f','\0'};
    char variables[input_dim];
    if (input_dim > 6){cout << "Support <7 variables!";};
    for (int i=0;i<input_dim;i++){
        variables[i] = variables2[i];
    }
    variables[input_dim] = '\0';
    //char variables[] = {'a','b','c','d','e','\0'};
    //cout << variables[0] << variables[1] << variables[2] << variables[3] << variables[4] << variables[5] << variables[6] << endl;
    // Note: the char array should end with '\0'
    char operations_1[] = {'>','<','~','\\','O','Q','I','\0'};
    char operations_2[] = {'+','*','-','/','\0'};
    ifstream input("../../data_samples/"+env+".txt");
    vector<vector<T>> data_var;
    vector<vector<T>> tangent_vecs;
    int test_pts_id = 0;
    for (string line; getline(input,line);)
    {
        if (test_pts_id == test_pts){break;}
        vector<T> dp_var;
        vector<T> dp_t;
        size_t pos = 0;
        string delimiter = " ";
        string token;
        int i = 0;
        while ((pos = line.find(delimiter)) != string::npos) {
            token = line.substr(0, pos);
            if (i < input_dim){
                dp_var.push_back(stof(token));
            }
            else {
                dp_t.push_back(stof(token));
            }
            line.erase(0, pos + delimiter.length());
            i = i + 1;
        }
        dp_t.push_back(stof(line));
        data_var.push_back(dp_var);
        tangent_vecs.push_back(dp_t);
        test_pts_id = test_pts_id + 1;
    }
    /*read arity file*/
    ifstream input2("./arity2templates.txt");
    vector<string> arities;
    /*number of templates tried*/
    int num_templates = 539;//89;
    int template_id = 0;
    for (string line; getline(input2,line);)
    {
        if (template_id == num_templates){break;}
        arities.push_back(line);
        template_id = template_id + 1;
    }
    vector<string> fail_arities = {};
    int num_cq = 0;
    int total_cq = 100;
    vector<vector<T>> test_vecs;
    vector<vector<T>> orth_bases;
    //vector<T> success_eps = {};
    //ofstream f("./success_eps.txt");
    /*for(vector<T>::const_iterator i = success_eps.begin(); i != success_eps.end(); ++i) {
        f << *i << '\n';
    }*/
    for (int k=0;k<num_templates;k++){

        string arity = arities[k]; //"0021"
        cout << "k=" << k << "," << "arity=" << arity << endl;
        /*string arity = "01012";*/
        /*string arity = "002";*/
        int arity_len = arity.length();

        if (arity.substr(arity_len-1,1) == "1") {
        cout << "skip(end with 1)" << endl ;continue;
        } else {
        cout << "search..." << endl;
        }

        vector<int> dims;
        vector<int> prod_dims;
        vector<string> symbol_types;
        int flat_size = 1;
        for (int i=0; i<arity.length();i++){
            string symbol_type = arity.substr(i,1);
            int dim = strlen(ops(symbol_type, variables, operations_1,operations_2));
            //cout << dim;
            symbol_types.push_back(symbol_type);
            dims.push_back(dim);
            prod_dims.push_back(flat_size);
            flat_size = flat_size * dim;
        }
        //cout << endl << flat_size << endl;

        for (int i=0; i<flat_size;i++){
            vector<int> each_id = each_ids(i, prod_dims);
            vector<char> RPN;
            for (int j=0;j<arity_len;j++){
                char* ops_ = ops(arity.substr(j,1), variables, operations_1, operations_2);
                char symbol = ops_[each_id[arity_len-1-j]];
                RPN.push_back(symbol);
            }
            /*vector<char> RPN = {'a','d','*','b','c','*','-'};*/
            /*vector<char> RPN = {'c','Q','d','Q','+'};*/
            //cout << i << endl;
            //printrpn(RPN);
            T eps = 0.0;
            bool flag = true;
            for (int j=0; j<test_pts; j++){
                vector<T> grad = H_grad_evaluate(RPN, arity_len, data_var[j], input_dim);
                vector<T> normalized_grad = normalized_vec(grad, input_dim);
                eps = eps*j + sqrt(pow(my_inner_product(normalized_grad, tangent_vecs[j], input_dim),2));
                //catch (int e) {eps=10.0;};
                eps = eps/(j+1);
                // eps cannot be -nan
                if (eps>0.000001 || isnan(eps)){flag=false; break;}
                }
            if (flag==true){
                //printrpn(RPN);
                //cout << endl << eps << endl;
                //cout << num_cq;
                bool independence = true;
                if (num_cq == 0){
                    /*cout << "cq:0" << endl;
                    printrpn(RPN);
                    /* generate dataset for checking independence */
                    for (int j=0; j<test_pts; j++){
                        vector<T> grad = H_grad_evaluate(RPN, arity_len, data_var[j], input_dim);
                        vector<T> normalized_grad = normalized_vec(grad, input_dim);
                        vector<T> rv = random_vec(input_dim);
                        vector<T> test_vec = project_orth(rv, normalized_grad, input_dim);
                        orth_bases.push_back(normalized_grad);
                        test_vecs.push_back(test_vec);
                        }
                    //printvec(orth_bases[0]);
                    }
                else {
                    /*cout << "cq>0" << endl;
                    printrpn(RPN);
                    cout << eps << endl;
                    /* check independence. Dependence <-> test_vecs and grad always orthogonal */
                    T eps2 = 0.0;
                    for (int j=0; j<test_pts; j++){
                        vector<T> grad = H_grad_evaluate(RPN, arity_len, data_var[j], input_dim);
                        vector<T> normalized_grad = normalized_vec(grad, input_dim);
                        eps2 = eps2*j + sqrt(pow(my_inner_product(normalized_grad, test_vecs[j], input_dim),2));
                        //catch (int e) {eps=10.0;};
                        eps2 = eps2/(j+1);
                        }
                        //cout << eps << endl;
                    if (eps2<0.0001){independence=false;}
                    else {
                    /* if survive, update the dataset (orth_bases & test_vecs) for independence check */
                    for (int j=0; j<test_pts; j++){
                        vector<T> grad = H_grad_evaluate(RPN, arity_len, data_var[j], input_dim);
                        vector<T> normalized_grad = normalized_vec(grad, input_dim);
                        //printvec(orth_bases[j]);
                        //cout << endl;
                        for (int k=0; k<input_dim; k++){
                            orth_bases[j].push_back(normalized_grad[k]);
                        }
                        //printvec(orth_bases[j]);
                        cout << endl;
                        vector<T> temp = schmidt(orth_bases[j], input_dim);
                        vector<T> rv = random_vec(input_dim);
                        for (int k=0; k<input_dim; k++){
                            orth_bases[j][k] = temp[k];
                            vector<T> test_vec = project_orth(rv, orth_bases[j], input_dim);
                            test_vecs[j][k] = test_vec[k];
                        }
                        //printvec(test_vecs[j]);
                    }
                    }
                }
                if (independence==true) {
                    num_cq = num_cq + 1;
                    /* This is an independent CQ, add to list */
                    string RPN_string(RPN.begin(), RPN.end());
                    //cout << RPN_string + " " + to_string(eps) <<endl ;
                    ofstream myfile;
                    myfile.open ("./winners_"+env+".txt", std::ios_base::app);
                    myfile << RPN_string + " " + to_string(eps) <<endl;
                    myfile.close();
                    if (num_cq == total_cq) {return 0;}
                }
            }
        }
    }
    return 0;
}
