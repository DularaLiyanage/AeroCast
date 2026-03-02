import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/aqi_prediction.dart';
import '../utils/constants.dart';

class AqiChart extends StatefulWidget {
  final List<AqiPrediction> predictions;

  const AqiChart({super.key, required this.predictions});

  @override
  State<AqiChart> createState() => _AqiChartState();
}

class _AqiChartState extends State<AqiChart> {
  int? _touchedIndex;

  @override
  Widget build(BuildContext context) {
    if (widget.predictions.isEmpty) return const SizedBox.shrink();

    // Calculate Y-axis bounds
    double maxY = 0;
    double minY = 500;
    for (var p in widget.predictions) {
      if (p.max > maxY) maxY = p.max;
      if (p.min < minY) minY = p.min;
    }
    maxY += 10;
    minY = (minY - 10).clamp(0, 500);

    final lineBarsData = [
      // Max Line (Hidden)
      LineChartBarData(
        spots: widget.predictions.asMap().entries.map((e) {
          return FlSpot(e.key.toDouble(), e.value.max);
        }).toList(),
        isCurved: true,
        color: Colors.transparent,
        dotData: FlDotData(show: false),
        belowBarData: BarAreaData(show: false),
      ),
      // Min Line (Hidden)
      LineChartBarData(
        spots: widget.predictions.asMap().entries.map((e) {
          return FlSpot(e.key.toDouble(), e.value.min);
        }).toList(),
        isCurved: true,
        color: Colors.transparent,
        dotData: FlDotData(show: false),
        belowBarData: BarAreaData(show: false),
      ),
      // Main AQI Line
      LineChartBarData(
        spots: widget.predictions.asMap().entries.map((e) {
          return FlSpot(e.key.toDouble(), e.value.aqi);
        }).toList(),
        isCurved: true,
        color: AppColors.primaryText,
        barWidth: 3,
        isStrokeCapRound: true,
        dotData: FlDotData(
            show: true,
            getDotPainter: (spot, percent, barData, index) {
              return FlDotCirclePainter(
                radius: 2,
                color: AppColors.primaryText,
                strokeWidth: 0,
              );
            }),
        belowBarData: BarAreaData(
          show: true,
          gradient: LinearGradient(
            colors: [
              AppColors.primaryText.withOpacity(0.1),
              AppColors.primaryText.withOpacity(0.0),
            ],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
      ),
    ];

    return AspectRatio(
      aspectRatio: 1.5,
      child: Padding(
        padding:
            const EdgeInsets.only(right: 16.0, left: 0, top: 10, bottom: 0),
        child: LineChart(
          LineChartData(
            lineTouchData: LineTouchData(
              touchTooltipData: LineTouchTooltipData(
                tooltipBgColor: Colors.black,
                getTooltipItems: (touchedSpots) {
                  return touchedSpots.map((spot) {
                    if (spot.barIndex != 2) return null;
                    final data = widget.predictions[spot.x.toInt()];
                    return LineTooltipItem('${data.time}\n',
                        const TextStyle(color: Colors.white70, fontSize: 12),
                        children: [
                          TextSpan(
                            text: '${data.aqi.toInt()} AQI',
                            style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                                fontSize: 14),
                          ),
                          TextSpan(
                            text: '\n${data.riskLevel}',
                            style: TextStyle(
                                color: AppColors.getColor(data.colour),
                                fontWeight: FontWeight.w600,
                                fontSize: 12),
                          )
                        ]);
                  }).toList();
                },
              ),
              touchCallback: (FlTouchEvent event, LineTouchResponse? response) {
                if (response == null || response.lineBarSpots == null) {
                  return;
                }
                // Handle Trace/Drag or Tap to update position
                if (event is FlTapDownEvent ||
                    event is FlPanDownEvent ||
                    event is FlPanUpdateEvent) {
                  if (response.lineBarSpots!.isNotEmpty) {
                    final spot = response.lineBarSpots!.first;
                    if (_touchedIndex != spot.x.toInt()) {
                      setState(() {
                        _touchedIndex = spot.x.toInt();
                      });
                    }
                  }
                }
              },
              handleBuiltInTouches:
                  false, // Disable default auto-dismiss behavior
            ),
            // Show tooltip permanently if tapped
            showingTooltipIndicators: _touchedIndex != null
                ? [
                    ShowingTooltipIndicators([
                      LineBarSpot(lineBarsData[2], 2,
                          lineBarsData[2].spots[_touchedIndex!])
                    ])
                  ]
                : [],
            gridData: FlGridData(
              show: true,
              drawVerticalLine: false,
              horizontalInterval: 20,
              getDrawingHorizontalLine: (value) => FlLine(
                color: Colors.grey.withOpacity(0.1),
                strokeWidth: 1,
              ),
            ),
            titlesData: FlTitlesData(
              show: true,
              rightTitles:
                  AxisTitles(sideTitles: SideTitles(showTitles: false)),
              topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
              bottomTitles: AxisTitles(
                sideTitles: SideTitles(
                  showTitles: true,
                  reservedSize: 30,
                  interval: 6,
                  getTitlesWidget: (value, meta) {
                    int index = value.toInt();
                    if (index >= 0 && index < widget.predictions.length) {
                      return Padding(
                        padding: const EdgeInsets.only(top: 8.0),
                        child: Text(
                          widget.predictions[index].time,
                          style: GoogleFonts.poppins(
                            color: Colors.grey,
                            fontSize: 10,
                          ),
                        ),
                      );
                    }
                    return const Text('');
                  },
                ),
              ),
              leftTitles: AxisTitles(
                sideTitles: SideTitles(
                  showTitles: true,
                  interval: 20,
                  reservedSize: 40,
                  getTitlesWidget: (value, meta) {
                    return Text(
                      value.toInt().toString(),
                      style: GoogleFonts.poppins(
                        color: Colors.grey,
                        fontSize: 10,
                      ),
                    );
                  },
                ),
              ),
            ),
            borderData: FlBorderData(show: false),
            minX: 0,
            maxX: widget.predictions.length.toDouble() - 1,
            minY: minY,
            maxY: maxY,
            lineBarsData: lineBarsData,
            betweenBarsData: [
              BetweenBarsData(
                fromIndex: 0,
                toIndex: 1,
                color: Colors.blueGrey.withOpacity(0.15),
              )
            ],
          ),
        ),
      ),
    );
  }
}
