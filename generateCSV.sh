echo "filename,label" > allTrain.csv
echo "filename,label" > allTest.csv
for out in $(find xrayTrain/ -name *.png | sort -R )
do
	LABEL=$(echo $out | cut -d "/" -f 2 | cut -d "_" -f3)
    	LABEL=$(echo "scale=3;$LABEL/1000" | bc -l)
	echo $out,$LABEL >> allTrain.csv
done

for out in $(find xrayTest/ -name *.png | sort -R )
do
        LABEL=$(echo $out | cut -d "/" -f 2 | cut -d "_" -f3)
        LABEL=$(echo "scale=3;$LABEL/1000" | bc -l)
        echo $out,$LABEL >> allTest.csv
done
