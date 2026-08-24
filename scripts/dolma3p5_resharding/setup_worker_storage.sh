#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: setup_worker_storage.sh --check|--apply [--layout auto|single|raid0]

Prepare EC2 NVMe instance-store devices for Dolma resharding.

  --check  Print detected devices and the direct-mount/RAID0 plan. No writes.
  --apply  Format only unused EC2 instance-store devices and mount /mnt/dolma.
  --layout Storage layout (default: auto). Auto selects a single device when
           only one exists and requires an explicit choice for multiple devices.

Environment overrides:
  DOLMA_WORK_MOUNT          Mount point (default: /mnt/dolma)
  DOLMA_MIN_AVAILABLE_BYTES Minimum required free bytes (default: 1650000000000)
  DOLMA_STORAGE_LAYOUT      Default value for --layout
EOF
}

mode=""
layout="${DOLMA_STORAGE_LAYOUT:-auto}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check|--apply)
      if [[ -n "$mode" ]]; then
        echo "Specify exactly one of --check or --apply" >&2
        exit 2
      fi
      mode="$1"
      ;;
    --layout)
      if [[ $# -lt 2 ]]; then
        echo "--layout requires auto, single, or raid0" >&2
        exit 2
      fi
      layout="$2"
      shift
      ;;
    --layout=*) layout="${1#*=}" ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

if [[ -z "$mode" ]]; then
  usage >&2
  exit 2
fi
case "$layout" in
  auto|single|raid0) ;;
  *) echo "--layout must be auto, single, or raid0" >&2; exit 2 ;;
esac

mount_point="${DOLMA_WORK_MOUNT:-/mnt/dolma}"
minimum_available_bytes="${DOLMA_MIN_AVAILABLE_BYTES:-1650000000000}"
marker_path="$mount_point/.dolma3p5-instance-store"

if ! [[ "$minimum_available_bytes" =~ ^[1-9][0-9]*$ ]]; then
  echo "DOLMA_MIN_AVAILABLE_BYTES must be a positive integer" >&2
  exit 2
fi

instance_devices=()
while read -r device device_type; do
  [[ "$device_type" == "disk" ]] || continue
  model=$(lsblk -dno MODEL "$device" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if [[ "$model" == *"EC2 NVMe Instance Storage"* ]]; then
    instance_devices+=("$device")
  fi
done < <(lsblk -dpno NAME,TYPE)

if [[ ${#instance_devices[@]} -eq 0 ]]; then
  echo "No EC2 NVMe instance-store devices were detected; refusing to continue" >&2
  exit 1
fi

echo "Detected ${#instance_devices[@]} EC2 instance-store device(s):"
instance_device_sizes=()
total_bytes=0
largest_device_index=0
largest_device_bytes=0
for device_index in "${!instance_devices[@]}"; do
  device="${instance_devices[$device_index]}"
  size_bytes=$(lsblk -bdno SIZE "$device")
  instance_device_sizes+=("$size_bytes")
  total_bytes=$((total_bytes + size_bytes))
  if (( size_bytes > largest_device_bytes )); then
    largest_device_index=$device_index
    largest_device_bytes=$size_bytes
  fi
done

# Add two percent for filesystem overhead when comparing raw device capacity
# with the requested usable capacity.
minimum_direct_device_bytes=$((minimum_available_bytes + minimum_available_bytes / 50))
layout_decision_required=0
case "$layout" in
  single)
    selected_devices=("${instance_devices[$largest_device_index]}")
    selected_raw_bytes=$largest_device_bytes
    ;;
  raid0)
    if [[ ${#instance_devices[@]} -lt 2 ]]; then
      echo "RAID0 requires at least two EC2 instance-store devices" >&2
      exit 1
    fi
    selected_devices=("${instance_devices[@]}")
    selected_raw_bytes=$total_bytes
    ;;
  auto)
    if [[ ${#instance_devices[@]} -eq 1 ]]; then
      layout="single"
      selected_devices=("${instance_devices[0]}")
      selected_raw_bytes=$largest_device_bytes
    else
      layout_decision_required=1
      selected_devices=("${instance_devices[@]}")
      selected_raw_bytes=$total_bytes
    fi
    ;;
esac

if (( layout_decision_required )); then
  if (( total_bytes < minimum_direct_device_bytes )); then
    echo "Combined instance-store capacity is too small for $minimum_available_bytes usable bytes" >&2
    exit 1
  fi
elif (( selected_raw_bytes < minimum_direct_device_bytes )); then
  echo "The selected $layout layout is too small for $minimum_available_bytes usable bytes" >&2
  exit 1
fi

for device_index in "${!instance_devices[@]}"; do
  device="${instance_devices[$device_index]}"
  model=$(lsblk -dno MODEL "$device" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if (( layout_decision_required )); then
    role="candidate"
  else
    role="unused"
    for selected_device in "${selected_devices[@]}"; do
      if [[ "$device" == "$selected_device" ]]; then
        role="selected"
        break
      fi
    done
  fi
  printf '  %s  %s bytes  %s  %s\n' \
    "$device" "${instance_device_sizes[$device_index]}" "$role" "$model"
done

hazards=0
for device in "${selected_devices[@]}"; do
  if lsblk -nrpo NAME,TYPE "$device" | tail -n +2 | grep -q .; then
    echo "    BLOCKED $device: device has child partitions or mappings" >&2
    hazards=1
  fi
  if lsblk -nrpo MOUNTPOINT "$device" | grep -q '[^[:space:]]'; then
    echo "    BLOCKED $device: device or a child is already mounted" >&2
    hazards=1
  fi
  if sudo wipefs -n "$device" | grep -q .; then
    echo "    BLOCKED $device: device already contains a filesystem or RAID signature" >&2
    hazards=1
  fi
done

if findmnt -rn "$mount_point" >/dev/null 2>&1; then
  mounted_source=$(findmnt -rn -o SOURCE "$mount_point")
  echo "$mount_point is already mounted from $mounted_source"
  if [[ ! -f "$marker_path" ]]; then
    echo "BLOCKED: existing mount is not marked as managed by this script" >&2
    exit 1
  fi
  available_bytes=$(df -B1 --output=avail "$mount_point" | tail -n 1 | tr -d ' ')
  if (( available_bytes < minimum_available_bytes )); then
    echo "BLOCKED: only $available_bytes bytes are available at $mount_point" >&2
    exit 1
  fi
  echo "Storage is already prepared with $available_bytes available bytes"
  exit 0
fi

if (( layout_decision_required )); then
  echo "Layout decision required for this multi-device worker:"
  if (( largest_device_bytes >= minimum_direct_device_bytes )); then
    echo "  --layout single: use ${instance_devices[$largest_device_index]} ($largest_device_bytes bytes raw)"
  else
    echo "  --layout single: insufficient capacity ($largest_device_bytes bytes raw)"
  fi
  echo "  --layout raid0: stripe ${#instance_devices[@]} devices ($total_bytes bytes raw) for aggregate local I/O"
elif [[ "$layout" == "single" ]]; then
  echo "Plan: format ${selected_devices[0]} as XFS and mount it at $mount_point"
else
  echo "Plan: create RAID0 across ${#selected_devices[@]} devices ($selected_raw_bytes bytes raw), format as XFS, and mount at $mount_point"
fi

if (( hazards )); then
  echo "Refusing to format instance-store devices because the checks above found existing state" >&2
  exit 1
fi

if (( layout_decision_required )) && [[ "$mode" == "--apply" ]]; then
  echo "Refusing to apply an implicit layout; rerun with --layout single or --layout raid0" >&2
  exit 1
fi

if [[ "$mode" == "--check" ]]; then
  echo "Check complete; no storage changes were made"
  exit 0
fi

if command -v dnf >/dev/null 2>&1; then
  package_manager=(sudo dnf install -y)
elif command -v yum >/dev/null 2>&1; then
  package_manager=(sudo yum install -y)
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  package_manager=(sudo apt-get install -y)
else
  echo "No supported package manager found" >&2
  exit 1
fi

"${package_manager[@]}" xfsprogs
sudo mkdir -p "$mount_point"

if [[ "$layout" == "single" ]]; then
  storage_device="${selected_devices[0]}"
else
  "${package_manager[@]}" mdadm
  storage_device=/dev/md0
  if [[ -e "$storage_device" ]]; then
    echo "$storage_device already exists; refusing to replace it" >&2
    exit 1
  fi
  sudo mdadm \
    --create "$storage_device" \
    --level=0 \
    --raid-devices="${#selected_devices[@]}" \
    --metadata=1.2 \
    --run \
    "${selected_devices[@]}"
fi

sudo mkfs.xfs "$storage_device"
sudo mount "$storage_device" "$mount_point"
sudo chown "$(id -u):$(id -g)" "$mount_point"
mkdir -p "$mount_point/dolma3p5-resharding"
{
  printf 'storage_device=%s\n' "$storage_device"
  printf 'instance_devices=%s\n' "${selected_devices[*]}"
} > "$marker_path"

available_bytes=$(df -B1 --output=avail "$mount_point" | tail -n 1 | tr -d ' ')
if (( available_bytes < minimum_available_bytes )); then
  echo "Prepared storage has only $available_bytes available bytes; minimum is $minimum_available_bytes" >&2
  exit 1
fi

echo "Prepared $mount_point with $available_bytes available bytes"
df -h "$mount_point"
