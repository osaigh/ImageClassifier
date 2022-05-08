import predict_utility as pu
import input_args as ia
import json
import numpy as np

def main():
    in_arg = ia.get_predict_input_args()

    if in_arg.input and in_arg.checkpoint:
        use_gpu = False
        if in_arg.gpu:
            use_gpu = True
        
        top_k = 5
        if in_arg.top_k:
            top_k = in_arg.top_k

        category_names = "NULL"
        if in_arg.category_names:
            category_names = in_arg.category_names
        
        #predict
        tp,tc = pu.predict(in_arg.input,in_arg.checkpoint,use_gpu = use_gpu, topk = top_k)

        top_class_list = list()

        if category_names != 'NULL':
            with open(category_names, 'r') as f:
                cat_to_name = json.load(f)

                for category in tc:
                    top_class_list.append(cat_to_name[str(category)])
                
        top_class = np.array(top_class_list)
        
        #print results
        print("Prediction Results ")
        print('%-8s%-30s%-30s' % ('#', 'Class','Probability'))
        for i in range(top_k):
            print('%-8i%-30s%-30s' % (i+1, top_class[i],tp[i]))

        return  top_class, tp  

    else:
        raise ValueError('image path and checkpoint are required')
    
if __name__ == "__main__":
    main()