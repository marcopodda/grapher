export CUDA_VISIBLE_DEVICES=2

python manage.py baseline ER community --metric=degree
python manage.py baseline ER community --metric=clustering
python manage.py baseline ER ego --metric=degree
python manage.py baseline ER ego --metric=clustering
python manage.py baseline ER ladders --metric=degree
python manage.py baseline ER ladders --metric=clustering
python manage.py baseline ER ENZYMES --metric=degree
python manage.py baseline ER ENZYMES --metric=clustering
python manage.py baseline ER PROTEINS_full --metric=degree
python manage.py baseline ER PROTEINS_full --metric=clustering
python manage.py baseline BA community --metric=degree
python manage.py baseline BA community --metric=clustering
python manage.py baseline BA ego --metric=degree
python manage.py baseline BA ego --metric=clustering
python manage.py baseline BA ladders --metric=degree
python manage.py baseline BA ladders --metric=clustering
python manage.py baseline BA ENZYMES --metric=degree
python manage.py baseline BA ENZYMES --metric=clustering
python manage.py baseline BA PROTEINS_full --metric=degree
python manage.py baseline BA PROTEINS_full --metric=clustering
python manage.py graphrnn community
python manage.py graphrnn ego
python manage.py graphrnn ladders
python manage.py graphrnn ENZYMES
python manage.py graphrnn PROTEINS

