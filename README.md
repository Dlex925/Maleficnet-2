More or less a log for me at this moment

Last command(s):

python maleficnet.py --epochs 0 --model vgg16 --payload payload_small.exe.payload --gamma 0.00001 --dataset imagenet --num_classes 1000 --dim 224 --only_pretrained  
python maleficnet.py --epochs 0 --model vgg16 --payload payload_small.exe.payload --gamma 0.00001 --dataset imagenet --num_classes 1000 --dim 224 --only_pretrained --hessian

Results(as of latest successful tests) :

Before injection:
test_acc            0.7110000252723694
test_loss            1.180359125137329

After random injection:

test_acc            0.7107999920845032
test_loss            1.180272102355957


After injection with hessian:

test_acc            0.7110000252723694
test_loss            1.1803901195526123


The base acc might be lower(71 instead of 73) because I used vgg, not vgg_bn in testing

To do:

Try with bigger payloads.
Retry with vgg_bn
Logging
