#ifndef CUTE_STYLE_RVIZ_PLUGIN__CUTE_STYLE_PANEL_HPP
#define CUTE_STYLE_RVIZ_PLUGIN__CUTE_STYLE_PANEL_HPP

#include <memory>
#include <string>

#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/config.hpp>
#include <rviz_common/panel.hpp>
#include <std_msgs/msg/string.hpp>

namespace cute_style_rviz_plugin
{

class CuteStylePanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit CuteStylePanel(QWidget * parent = nullptr);

protected:
  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

private Q_SLOTS:
  void applyDefaultTheme();
  void applyThemeFromFile();
  void browseThemeFile();
  void resetTheme();
  void toggleTopicSubscription(int state);
  void topicNameEdited();

private:
  void ensureRosNode();
  void ensureThemeSubscription();
  void dropThemeSubscription();

  void setStatus(const QString & text);
  void applyStyleSheet(const QString & qss, const QString & source);
  QString loadTextFile(const QString & path, QString & error) const;
  QString defaultThemeQss() const;

  rclcpp::Node::SharedPtr ros_node_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr theme_sub_;

  std::string theme_topic_name_;

  QLabel * title_label_;
  QLabel * status_label_;

  QLineEdit * theme_path_edit_;
  QPushButton * browse_button_;
  QPushButton * apply_file_button_;
  QPushButton * apply_default_button_;
  QPushButton * reset_button_;

  QCheckBox * subscribe_checkbox_;
  QLineEdit * topic_name_edit_;
};

}  // namespace cute_style_rviz_plugin

#endif  // CUTE_STYLE_RVIZ_PLUGIN__CUTE_STYLE_PANEL_HPP
