import time
from pathlib import Path
from app.services.horse_cascade_service import HorseCascadeService

def main():
    service = HorseCascadeService(
        detector_path="/home/maks/Moderator-CNN/models/horse/yolov8_horse_head.pt",
        sam_path="/home/maks/Moderator-CNN/models/horse/mobile_sam.pt",
        top_k=5,
        match_threshold=15
    )

    val_dir = Path("/home/maks/Moderator-CNN/data/Horse")
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([p for p in val_dir.rglob("*") if p.suffix.lower() in valid_exts])

    print(f"🚀 [Индексация] Загрузка галереи снимков из {val_dir}...")
    enrolled_count = 0
    gallery_paths = []
    query_paths = []

    # Разделяем датасет: 1-е фото каждой лошади -> в галерею, остальные -> тестовые запросы
    seen_horses = set()
    for p in image_paths:
        horse_id = p.parent.name
        if horse_id not in seen_horses:
            if service.enroll_horse(horse_id, str(p)):
                seen_horses.add(horse_id)
                enrolled_count += 1
        else:
            query_paths.append((horse_id, str(p)))

    print(f"✅ В галерею успешно зарегистрировано особей: {enrolled_count}")
    print(f"🧪 Запуск поиска 1:N для {len(query_paths)} контрольных запросов...\n")

    correct_matches = 0
    false_matches = 0
    missed_matches = 0
    total_time = 0.0

    for true_id, q_path in query_paths:
        t0 = time.time()
        result = service.search_1_to_n(q_path)
        elapsed = (time.time() - t0) * 1000
        total_time += elapsed

        status = result["status"]
        matched_id = result["matched_horse_id"]
        inliers = result["max_inliers"]

        if status == "MATCH":
            if matched_id == true_id:
                correct_matches += 1
                tag = "✅ CORRECT"
            else:
                false_matches += 1
                tag = "❌ FALSE MATCH"
        else:
            missed_matches += 1
            tag = "⚠️ MISSED (NO MATCH)"

        print(f"  Query ID: {true_id:10s} | Match ID: {str(matched_id):10s} | Inliers: {inliers:2d} | Latency: {elapsed:5.1f} ms | {tag}")

    avg_latency = total_time / max(1, len(query_paths))
    print("\n=== ИТОГИ КАСКАДНОГО ПОИСКА 1:N ===")
    print(f"  • Всего тестовых запросов : {len(query_paths)}")
    print(f"  • Верных распознаваний    : {correct_matches}")
    print(f"  • Ложных срабатываний (FAR): {false_matches}")
    print(f"  • Пропущенных (FRR)        : {missed_matches}")
    print(f"  • Средняя задержка поиска : {avg_latency:.2f} ms")

if __name__ == "__main__":
    main()