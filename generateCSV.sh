echo "filename,label" > allTrain.csv
echo "filename,label" > allVal.csv
echo "filename,label" > allTest.csv
for out in $(find xrayTrain/ -name *.png | sort -R )
do
	LABEL=$(echo $out | cut -d "/" -f 2 | cut -d "_" -f3)
    	LABEL=$(echo "scale=3;$LABEL/1000" | bc -l)
	echo $out,$LABEL >> allTrain.csv
done

LINES=$(cat allTrain.csv | wc -l)

echo $LINES


for out in $(find xrayTest/ -name *.png | sort -R )
do
        LABEL=$(echo $out | cut -d "/" -f 2 | cut -d "_" -f3)
        LABEL=$(echo "scale=3;$LABEL/1000" | bc -l)
        echo $out,$LABEL >> allTest.csv
done

split -l $(echo "4368*0.8" | bc -l | cut -d"." -f1) allTrain.csv

mv xaa allTrain.csv
mv xab allVal.csv

