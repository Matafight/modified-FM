import matplotlib.pyplot as plt
import draw as my_draw

if '__name__ == __main__':
    md = my_draw.draw_result()
    ord_test = 'test_12-26-12-41_1.0__1.0_k_10_u4.base.txt'
    ord_train = 'train_12-26-12-41_1.0__1.0_k_10_u4.base.txt'
    sgl_test = 'test_12-26-12-43_0.001__0.001_k_10_u4.base.txt'
    sgl_train = 'train_12-26-12-43_0.001__0.001_k_10_u4.base.txt'
    test_dir_sgl = './sgl-fm-fix-share/results/'+sgl_test
    test_dir_ord = './ord-fm-fix-share/results/'+ord_test
 
    train_dir_sgl = "./sgl-fm-fix-share/results/"+sgl_train
    train_dir_ord = "./ord-fm-fix-share/results/"+ord_train
    md.draw_test(train_dir_ord,train_dir_sgl,iftest = False)
