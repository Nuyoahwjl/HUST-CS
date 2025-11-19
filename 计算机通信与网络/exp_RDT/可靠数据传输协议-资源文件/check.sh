#!/bin/sh

# 用法：
#     ./check.sh -x 可执行文件 -i 输入文件
usage() {
    echo "Usage: $0 -x {GBN_PATH} -i {INPUT_FILE}"
}

while getopts ":x:i:" o; do
    case "${o}" in
        x)
            gbn_path="$OPTARG"
            ;;
        i)
            input_file="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "$gbn_path" ] || [ -z "$input_file" ]; then
    usage
    exit 1
fi

gbn_dir="$(dirname "$gbn_path")"
gbn="$(basename "$gbn_path")"

output_file='output.txt'
result_file="$(realpath --relative-to="$gbn_dir" 'result.txt')"

for i in $(seq 10); do
    echo "GBN Test $i:"
    # 防止你在 GBN 中对输入文件路径进行了硬编码
    (
        cd "$gbn_dir" || exit 1
        "./$gbn" > "$result_file" 2>&1
    )
    diff "$input_file" "$output_file"
done
