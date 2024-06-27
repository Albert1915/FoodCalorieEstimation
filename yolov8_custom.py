from ultralytics import YOLO

# Memuat model YOLOv8
model = YOLO('model.pt')

# Memuat informasi kalori dari file calorie.txt
def load_calories_from_txt(txt_file):
    calorie_data = {}
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():  # Memastikan tidak ada baris kosong
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    class_name = parts[0].strip()
                    calorie_value = parts[1].strip()
                    calorie_data[class_name] = calorie_value
    return calorie_data

calorie_data = load_calories_from_txt('calorie.txt')

# Melakukan deteksi objek dari sumber video atau kamera
result = model(source=1, show=True, conf=0.3, save=True)

# Loop melalui hasil deteksi dan tampilkan informasi kalori di terminal log
for det in result.pred:
    class_id = det['class_ids'].item()
    class_name = model.names[class_id]
    confidence = det['scores'].item()
    
    # Ambil informasi kalori jika tersedia untuk kelas yang terdeteksi
    if class_name in calorie_data:
        calorie_info = calorie_data[class_name]
        
        # Simulasi perhitungan kalori berdasarkan area bounding box (contoh sederhana)
        bbox_width = det['box'][2].item()
        bbox_height = det['box'][3].item()
        area = bbox_width * bbox_height
        estimated_calories = int(calorie_info) * (area / 1000)  # Contoh sederhana untuk estimasi
        
        # Tambahkan informasi kalori ke dalam log
        calorie_log = f"Estimated Calories: {estimated_calories:.2f} kalori"
    else:
        calorie_log = 'Informasi kalori tidak tersedia'
    
    # Tampilkan informasi deteksi beserta kalori yang terkait di terminal log
    print(f"Detected: {class_name} with confidence: {confidence:.2f}, {calorie_log}")
