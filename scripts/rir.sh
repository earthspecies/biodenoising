RIR_PATH="/home/marius/data/biodenoising16k/rir"

mkdir -p $RIR_PATH

AZURE_URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"
BLOB="datasets_fullband.impulse_responses_000.tar.bz2"

URL="$AZURE_URL/$BLOB"
echo "Download: $BLOB"

# ### DRY RUN: print HTTP response and Content-Length
# # WITHOUT downloading the files
# curl -s -I "$URL" | head -n 2

### Actually download the files: UNCOMMENT when ready to download
# curl "$URL" -o "$OUTPUT_PATH/$BLOB"

### Same as above, but using wget
# wget "$URL" -O "$OUTPUT_PATH/$BLOB"

### Same, + unpack files on the fly
curl "$URL" | tar -C "$RIR_PATH" -f - -x -j